// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx906 (GCN5 / Vega20) kernel-dispatch overlays. Phase 2.

use super::super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// HFQ4 gate_up MMQ fused-projection kernel — gfx906 wave64. 2-way
    /// fused {a, b} on a single launch. Generic naming so the same
    /// entry serves BOTH gate_up (a=gate, b=up) and LA QKVZA-head
    /// (a=qkv, b=z) dispatch sites. See
    /// `kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh`.
    ///
    /// Caller invariants:
    ///   - m_a, m_b multiples of MMQ_Y(=128).
    ///   - batch_size ≥ should_use_mmq cutover (gfx906 default: 8).
    ///   - x is the raw fp16 activation (ensure_q8_1_mmq_x is called
    ///     internally; caller MAY pre-quantize via the prequant
    ///     sibling below if Xq is already on hand from another call
    ///     on the same x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq4g256_mmq_gfx906(
        &mut self,
        a_a: &GpuTensor,
        a_b: &GpuTensor,
        x: &GpuTensor,
        y_a: &GpuTensor,
        y_b: &GpuTensor,
        m_a: usize,
        m_b: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_gate_up_hfq4g256_mmq_gfx906_prequant(
            a_a, a_b, xq_ptr, y_a, y_b, m_a, m_b, k, batch_size,
        )
    }
    /// Pre-quantized X variant — caller passes the Q8_1 scratch pointer
    /// produced by an earlier `ensure_q8_1_mmq_x(x, batch_size, k)` call.
    /// Used by the LA QKVZA-head site, which has already quantized X
    /// when it computed the qkv/z 2-way and then continues to the
    /// β+α tail. Mirrors the `_prequant` sibling on the dp4a path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq4g256_mmq_gfx906_prequant(
        &mut self,
        a_a: &GpuTensor,
        a_b: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y_a: &GpuTensor,
        y_b: &GpuTensor,
        m_a: usize,
        m_b: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.should_use_mmq(batch_size) || self.capture_mode,
            "gate_up_hfq4g256_mmq_gfx906 called at non-winning B={} (capture={})",
            batch_size,
            self.capture_mode,
        );

        // MMQ_Y selection. Y=64 is the higher-occupancy variant (plan §6.5);
        // wrappers only exist for {x16, x32}, so larger mmq_x at y64 falls
        // back to y128. Y=128 is the established default (matches the
        // residual sibling).
        let y64 = self.flags.hfq4_mmq_gfx906_y64_enabled();
        let mmq_y: usize = if y64 { 64 } else { 128 };

        debug_assert!(
            m_a % mmq_y == 0 && m_b % mmq_y == 0,
            "gate_up_hfq4g256_mmq_gfx906 requires m_a/m_b multiples of MMQ_Y={mmq_y} (got a={m_a} b={m_b})",
        );

        let mut mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 32 {
            32
        } else {
            64
        };
        // Y=64 only has wrappers for x16 and x32; cap mmq_x at 32 when
        // y64 is requested. Falls through to the y128 path for tiny
        // batches (x8) since no x8_y64 wrapper exists.
        let use_y64 = y64 && mmq_x >= 16;
        if use_y64 && mmq_x > 32 {
            mmq_x = 32;
        }

        let is_full = m_a % mmq_y == 0 && m_b % mmq_y == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_gate_up_hfq4g256_mmq_gfx906";
        let y_suffix = if use_y64 { "_y64" } else { "" };
        let kernel_name = if is_full {
            format!("{}_full_set_x{}{}", base_name, mmq_x, y_suffix)
        } else {
            format!("{}_x{}{}", base_name, mmq_x, y_suffix)
        };

        let wrapper_src = match (mmq_x, use_y64) {
            (8, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X8_SRC,
            (16, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X16_SRC,
            (32, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X32_SRC,
            (64, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X64_SRC,
            (16, true) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X16_Y64_SRC,
            (32, true) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X32_Y64_SRC,
            _ => unreachable!("no gate_up wrapper for mmq_x={mmq_x} y64={use_y64}"),
        };
        let inlined = wrapper_src.replace(
            "#include \"gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh\"",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_BODY_CUH,
        );
        self.ensure_kernel(
            &format!("{}_x{}{}", base_name, mmq_x, y_suffix),
            &inlined,
            &kernel_name,
        )?;

        let mut a_a_p = a_a.buf.as_ptr();
        let mut a_b_p = a_b.buf.as_ptr();
        let mut xq = x_q8_ptr;
        let mut y_a_p = y_a.buf.as_ptr();
        let mut y_b_p = y_b.buf.as_ptr();
        let mut m_a_val = m_a as i32;
        let mut m_b_val = m_b as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_a_p as *mut _ as *mut c_void,
            &mut a_b_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_a_p as *mut _ as *mut c_void,
            &mut y_b_p as *mut _ as *mut c_void,
            &mut m_a_val as *mut _ as *mut c_void,
            &mut m_b_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        // KEEP IN SYNC WITH body.cuh: MMQ_Y is chosen by the wrapper
        // (`use_y64` above gates which wrapper is included). x_dm sizing
        // is MMQ_Y float2s. row_tiles uses the same MMQ_Y.
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        let x_dm_float2: usize = mmq_y;
        let total_m = m_a + m_b;
        let row_tiles = (total_m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;
        let shared_mem =
            ((mmq_y * x_stride * 4) + (x_dm_float2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 gate_up MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        let bytes = crate::profile::gemv_hfq4g256_bytes(m_a, k)
            + crate::profile::gemv_hfq4g256_bytes(m_b, k)
            + batch_size * k
            + batch_size * (m_a + m_b) * 4;
        // Distinct timer label per Y variant so attribution shows the split.
        let timer_label: &'static str = if use_y64 {
            "gemm_gate_up_hfq4g256_mmq_gfx906_y64"
        } else {
            "gemm_gate_up_hfq4g256_mmq_gfx906"
        };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", timer_label, bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, col_tiles as u32, 1],
            [64, 4, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_a_p);
                b.push_ptr(a_b_p);
                b.push_ptr(xq);
                b.push_ptr(y_a_p);
                b.push_ptr(y_b_p);
                b.push_i32(m_a_val);
                b.push_i32(m_b_val);
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
    /// HFQ4 qkv MMQ fused-projection kernel — gfx906 wave64. 3-way fused
    /// {Q, K, V} on a single launch, eliminating 2 of 3 launch overheads
    /// and amortizing L2 hits on the Q8_1 batch tile across the three
    /// outputs. See `kernels/src/gemm_qkv_hfq4g256_mmq_gfx906_body.cuh`
    /// for the kernel design and
    /// `docs/plans/experiments-archive/gfx906-fused-mmq-probe-results.md` for the §6.1
    /// probe that motivated this work.
    ///
    /// Caller invariants:
    ///   - q_m, k_m, v_m must each be multiples of MMQ_Y(=128). Qwen3.5
    ///     family satisfies (9B: q_m=4096, k_m=v_m=1024; 4B: q_m=2048,
    ///     k_m=v_m=512).
    ///   - batch_size ≥ should_use_mmq cutover (gfx906 default: 8).
    ///   - x is the same activation tensor as the residual sibling
    ///     expects (raw fp16, ensure_q8_1_mmq_x is called internally).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq4g256_mmq_gfx906(
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
        debug_assert!(
            self.arch_caps.should_use_mmq(batch_size) || self.capture_mode,
            "qkv_hfq4g256_mmq_gfx906 called at non-winning B={} (capture={})",
            batch_size,
            self.capture_mode,
        );
        debug_assert!(
            q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0,
            "qkv_hfq4g256_mmq_gfx906 requires q_m/k_m/v_m multiples of MMQ_Y=128 (got q={q_m} k={k_m} v={v_m})",
        );

        // Same mmq_x sweep as the gfx906 single-output mmq_set path so
        // future MMQ_X tuning translates 1:1. Note: only the {8,16,32,64}
        // quartet is wired up initially (the most common batch buckets);
        // the in-between values fall up to the next available mmq_x.
        let mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 32 {
            32
        } else {
            64
        };

        let is_full = q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_qkv_hfq4g256_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_set_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X16_SRC,
            32 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X32_SRC,
            64 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        let inlined = wrapper_src.replace(
            "#include \"gemm_qkv_hfq4g256_mmq_gfx906_body.cuh\"",
            kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_BODY_CUH,
        );
        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh.
        // X_STRIDE varies with mmq_x (see body.cuh x_stride_for<>):
        //   mmq_x ≥ 32 → stride 40 (b128 path), mmq_x < 32 → stride 33 (b32).
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_FLOAT2: usize = 128;
        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;
        let shared_mem =
            ((MMQ_Y * x_stride * 4) + (X_DM_FLOAT2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 qkv MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        // Total bytes = weight read (Q+K+V) + X read (Q8_1) + Y writes (3 outputs).
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k
            + batch_size * (q_m + k_m + v_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_mmq_gfx906", bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, col_tiles as u32, 1],
            [64, 4, 1], // wave64 native: 4 wave64s = 256 threads
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(xq);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
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
}
