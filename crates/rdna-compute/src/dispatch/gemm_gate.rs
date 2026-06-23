// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fused gate/up MLP-projection GEMMs (all dtypes). Pure move (Phase 1 M6).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

impl Gpu {
    /// MQ4-Lloyd WMMA fused gate+up GEMM (FFN preamble). 2-way fused.
    /// Phase B1 sibling.
    pub fn gemm_gate_up_mq4g256_lloyd_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        let total_m = gate_m + up_m;
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
            return self.gemm_gate_up_mq4g256_lloyd_wmma_mb4(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, n,
            );
        }
        self.bind_thread()?;
        let (src, module) = kernels::gemm_gate_up_mq4g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_gate_up_mq4g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_mq4g256_lloyd_wmma",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_mq4g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
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
    /// Phase D-B: 16×64 fanout sibling of `gemm_gate_up_mq4g256_lloyd_wmma`.
    pub fn gemm_gate_up_mq4g256_lloyd_wmma_mb4(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_gate_up_mq4g256_lloyd_wmma_mb4_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "gemm_gate_up_mq4g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_mq4g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_mq4g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
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
    pub fn gemm_gate_up_mq3g256_lloyd_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = gate_m + up_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_gate_up_mq3g256_lloyd_wmma_mb4(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, n,
            );
        }
        let (src, module) = kernels::gemm_gate_up_mq3g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_gate_up_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_mq3g256_lloyd_wmma",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
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
    /// MQ3-Lloyd gate_up mb4 dispatch.
    pub fn gemm_gate_up_mq3g256_lloyd_wmma_mb4(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_gate_up_mq3g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_gate_up_mq3g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * super::LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_mq3g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_mq3g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
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
    /// Batched 2-way fused HFQ4-G256 GEMM for the FFN preamble (gate + up).
    ///
    /// Processes N tokens × both projections (w_gate + w_up) in one launch.
    /// Bitwise-identical to calling `fused_gate_up_hfq4g256` N times on the
    /// same x[b] — 4-accumulator interleave + pairwise combine preserved
    /// per batch element.
    pub fn gemm_gate_up_hfq4g256(
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
        // CDNA3 MFMA path (task #130): two back-to-back rocBLAS calls against
        // the gate/up FP16 shadows. rocBLAS launch overhead is small compared
        // to the GEMM work at prefill batches, so fusing into a single
        // concatenated matrix isn't worth the extra kernel code tonight.
        let cdna3 = self.arch_caps.is_cdna3();
        if self.flags.hfq4_gate_up_fast
            && cdna3
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.capture_mode
        {
            if let Ok(Some(w_gate_ptr)) = self.ensure_fp16_shadow(a_gate, gate_m, k) {
                if let Ok(Some(w_up_ptr)) = self.ensure_fp16_shadow(a_up, up_m, k) {
                    let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                    let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                    let wgate = unsafe { DeviceBuffer::from_raw(w_gate_ptr, (gate_m * k) * 2) };
                    let wup = unsafe { DeviceBuffer::from_raw(w_up_ptr, (up_m * k) * 2) };
                    let gate_bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k);
                    let up_bytes = crate::profile::gemv_hfq4g256_bytes(up_m, k);
                    let timer = crate::profile::begin_timer(
                        &self.hip,
                        "gemm",
                        "gemm_gate_up_hfq4g256_rocblas",
                        gate_bytes + up_bytes,
                    );
                    let r1 = self.rocblas_gemm_hfq4_prefill(
                        &wgate,
                        &xb,
                        &y_gate.buf,
                        gate_m,
                        batch_size,
                        k,
                    );
                    let r2 = if r1.is_ok() {
                        self.rocblas_gemm_hfq4_prefill(&wup, &xb, &y_up.buf, up_m, batch_size, k)
                    } else {
                        Ok(())
                    };
                    std::mem::forget(xb);
                    std::mem::forget(wgate);
                    std::mem::forget(wup);
                    if let Some(t) = timer {
                        t.finish(&self.hip);
                    }
                    return r1.and(r2);
                }
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if self.flags.hfq4_gate_up_fast && batch_size > 1 && !self.flags.fp16_disabled {
            // gfx906 dp4a MMQ — default-on at batch_size ≥ 8 (per
            // should_use_mmq's gfx906 default). Quantize X once, screen
            // both weights, dispatch MMQ for each in set mode (add=0).
            // See docs/plans/gfx906-mmq-prd.md for context.
            let mut mmq_screen_rejected = false;
            if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_gate, gate_m, k)
                        && self.mmq_screen_weight(a_up, up_m, k)
                } else {
                    true
                };
                if use_mmq {
                    if gate_m % 128 == 0 && up_m % 128 == 0 {
                        return self.gemm_gate_up_hfq4g256_mmq_gfx906(
                            a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                        );
                    }
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self
                        .gemm_hfq4g256_mmq_set_gfx906(a_gate, xq, y_gate, gate_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_gfx906(a_up, xq, y_up, up_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    return r1.and(r2);
                }
                mmq_screen_rejected = self.mmq_screen;
                // screening rejected at least one weight — fall through; the
                // screen-reject path skips dp4a and lands on fp16 to preserve
                // the higher-precision fallback intent (dp4a shares the Q8_1
                // quant step that MMQ already failed on for this weight).
            }
            // gfx906 dp4a 2-way fused (issue #276 Gap 2). Fires for B>1
            // below the MMQ cutover or in capture mode. Skipped on
            // screen-reject.
            if !mmq_screen_rejected && self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
                return self.gemm_gate_up_hfq4g256_wave64_dp4a(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if self.arch_caps.is_gcn5_wave64() {
                return self.gemm_gate_up_hfq4g256_fp16_wave64(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            if self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen {
                    self.mmq_screen_weight(a_gate, gate_m, k)
                        && self.mmq_screen_weight(a_up, up_m, k)
                } else {
                    true
                };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self
                        .gemm_hfq4g256_mmq_set_prequant(a_gate, xq, y_gate, gate_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_up, xq, y_up, up_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    return r1.and(r2);
                }
            }
            // HFQ4 wave32 MMQ RDNA2 path (issue #299 Phase 3).
            if self.arch_caps.has_hfq4_mmq() && gate_m % 128 == 0 && up_m % 128 == 0 {
                return self.gemm_gate_up_hfq4g256_mmq(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // WMMA on gfx12 (RDNA4)
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_gate_up_hfq4g256_wmma_gfx12(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // WMMA on gfx11 (RDNA3)
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_gate_up_hfq4g256_wmma(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq4g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_gate_up_hfq4g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256",
            kernels::GEMM_GATE_UP_HFQ4G256_SRC,
            "gemm_gate_up_hfq4g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq4g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256", bytes);
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
    pub fn gemm_gate_up_hfq4g256_exact(
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
            "gemm_gate_up_hfq4g256",
            kernels::GEMM_GATE_UP_HFQ4G256_SRC,
            "gemm_gate_up_hfq4g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq4g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_exact", bytes);
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
    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `__ockl_fdot2`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    pub fn gemm_gate_up_hfq4g256_dot2(
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
            "gemm_gate_up_hfq4g256_dot2",
            kernels::GEMM_GATE_UP_HFQ4G256_DOT2_SRC,
            "gemm_gate_up_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_dot2"];

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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_dot2", bytes);
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
    /// Batched 2-way fused HFQ3-G256 GEMM for the FFN preamble (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_gate_up_hfq4g256` — single scalar variant only.
    /// Phase 1 of the gfx10 MQ3 prefill plan.
    pub fn gemm_gate_up_hfq3g256(
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
        // Phase 3 MMQ (auto-tile-selecting). Default-on for the supported
        // allowlist unless HIPFIRE_HFQ3_MMQ=0, and gate_m/up_m must be
        // MMQ_Y-aligned. Auto-selector falls back to dot2 at small batch.
        // Layer-gate is a no-op when unset (#302).
        if batch_size > 1
            && self.arch_caps.has_hfq3_mmq()
            && self.flags.hfq3_mmq_layer_gate_pass()
            && gate_m % 128 == 0
            && up_m % 128 == 0
        {
            return self.gemm_gate_up_hfq3g256_mmq(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        // Phase 2 experimental: wave32 dp4a if HIPFIRE_HFQ3_DP4A=1.
        if batch_size > 1 && self.arch_caps.has_hfq3_dp4a() {
            return self.gemm_gate_up_hfq3g256_dp4a(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        // FP16 fast paths — Phase 2b (dot2) + Phase 2c (fp16 fallback).
        // Layer-aware FP16 gate (#302).
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq3g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            return self.gemm_gate_up_hfq3g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256",
            kernels::GEMM_GATE_UP_HFQ3G256_SRC,
            "gemm_gate_up_hfq3g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq3g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256", bytes);
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
    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ3-G256 GEMM (gate + up).
    /// HFQ3 sibling of `gemm_gate_up_hfq4g256_dot2`. Phase 2b.
    pub fn gemm_gate_up_hfq3g256_dot2(
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
            "gemm_gate_up_hfq3g256_dot2",
            kernels::GEMM_GATE_UP_HFQ3G256_DOT2_SRC,
            "gemm_gate_up_hfq3g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq3g256_dot2"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_dot2", bytes);
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
    /// v_pk_fma_f16-accelerated batched 2-way fused HFQ3-G256 GEMM (gate + up).
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    pub fn gemm_gate_up_hfq3g256_fp16(
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
            "gemm_gate_up_hfq3g256_fp16",
            kernels::GEMM_GATE_UP_HFQ3G256_FP16_SRC,
            "gemm_gate_up_hfq3g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq3g256_fp16"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_fp16", bytes);
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
    /// Wave32+dp4a batched 2-way fused HFQ3-G256 GEMM (gate + up).
    /// Phase 2 experimental sibling of `gemm_qkv_hfq3g256_dp4a`.
    pub fn gemm_gate_up_hfq3g256_dp4a(
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
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq3g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            return self.gemm_gate_up_hfq3g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_dp4a",
            kernels::GEMM_GATE_UP_HFQ3G256_DP4A_SRC,
            "gemm_gate_up_hfq3g256_dp4a",
        )?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions["gemm_gate_up_hfq3g256_dp4a"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k
            + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_dp4a", bytes);
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
    /// HFQ3 gate_up MMQ auto-selector. Default-on unless `HIPFIRE_HFQ3_MMQ=0`.
    /// CALLER INVARIANT: gate_m and up_m must each be multiples of 128.
    pub fn gemm_gate_up_hfq3g256_mmq(
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
        // bind_thread: skip — delegates to gemm_gate_up_hfq3g256_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq3g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            return self.gemm_gate_up_hfq3g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        if batch_size <= 12 {
            self.gemm_gate_up_hfq3g256_dot2(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        } else if batch_size <= 127 {
            self.gemm_gate_up_hfq3g256_mmq_x16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        } else {
            self.gemm_gate_up_hfq3g256_mmq_x32(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        }
    }
    /// HFQ3 gate_up MMQ at mmq_x=8.
    pub fn gemm_gate_up_hfq3g256_mmq_x8(
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
        self.launch_gate_up_hfq3_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            8,
            "gemm_gate_up_hfq3g256_mmq_x8",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X8_SRC,
        )
    }
    /// HFQ3 gate_up MMQ at mmq_x=16.
    pub fn gemm_gate_up_hfq3g256_mmq_x16(
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
        self.launch_gate_up_hfq3_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            16,
            "gemm_gate_up_hfq3g256_mmq_x16",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X16_SRC,
        )
    }
    /// HFQ3 gate_up MMQ at mmq_x=32.
    pub fn gemm_gate_up_hfq3g256_mmq_x32(
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
        self.launch_gate_up_hfq3_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            "gemm_gate_up_hfq3g256_mmq_x32",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X32_SRC,
        )
    }
    /// HFQ3 gate_up MMQ mmq_x=32, MMQ_Y=96.
    pub fn gemm_gate_up_hfq3g256_mmq_x32_y96(
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
        self.launch_gate_up_hfq3_mmq_tile_with_y(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            96,
            "gemm_gate_up_hfq3g256_mmq_x32_y96",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X32_Y96_SRC,
        )
    }
    /// HFQ3 gate_up MMQ mmq_x=32, MMQ_Y=64.
    pub fn gemm_gate_up_hfq3g256_mmq_x32_y64(
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
        self.launch_gate_up_hfq3_mmq_tile_with_y(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            64,
            "gemm_gate_up_hfq3g256_mmq_x32_y64",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X32_Y64_SRC,
        )
    }
    pub fn gemm_gate_up_hfq4g256_mmq_x16(
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
        self.launch_gate_up_hfq4_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            16,
            "gemm_gate_up_hfq4g256_mmq_x16",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_X16_SRC,
        )
    }
    pub fn gemm_gate_up_hfq4g256_mmq_x32(
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
        self.launch_gate_up_hfq4_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            "gemm_gate_up_hfq4g256_mmq_x32",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_X32_SRC,
        )
    }
    pub fn gemm_gate_up_hfq4g256_mmq(
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
        if batch_size <= 63 {
            self.gemm_gate_up_hfq4g256_mmq_x16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        } else {
            self.gemm_gate_up_hfq4g256_mmq_x32(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        }
    }
    /// FP16-packed batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    pub fn gemm_gate_up_hfq4g256_fp16(
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
            "gemm_gate_up_hfq4g256_fp16",
            kernels::GEMM_GATE_UP_HFQ4G256_FP16_SRC,
            "gemm_gate_up_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_fp16"];

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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_fp16", bytes);
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
    /// GCN5 wave64 FP16 hybrid batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// block=[64,1,1] with 2 rows/block via warp_id. Halves grid.x vs wave32.
    /// Default-on for gfx906; gfx908 opts in via HIPFIRE_GCN5_WAVE64_HYBRID=1.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq4g256_fp16_wave64(
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
            "gemm_gate_up_hfq4g256_fp16_wave64",
            kernels::GEMM_GATE_UP_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_gate_up_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_fp16_wave64"];

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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq4g256_fp16_wave64",
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
    /// HFP4-G32 batched 2-way fused GEMM (gate + up). Routes gfx11/gfx12.
    pub fn gemm_gate_up_hfp4g32(
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
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_gate_up_hfp4g32_wmma_gfx12(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.gemm_gate_up_hfp4g32_wmma(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size)
    }
    pub fn gemm_gate_up_hfp4g32_wmma(
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
            "gemm_gate_up_hfp4g32_wmma",
            kernels::GEMM_GATE_UP_HFP4G32_WMMA_SRC,
            "gemm_gate_up_hfp4g32_wmma",
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
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfp4g32_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfp4g32_wmma",
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
    /// WMMA-accelerated batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_gate_up_hfq4g256_wmma(
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
        // HIPFIRE_GATE_UP_VARIANT=ldsx routes to the LDS-staged X variant
        // (Gate 1 microbench, opt-in only, default off). See
        // docs/perf-checkpoints/2026-05-01-gate-up-lds-x-share-plan.md.
        let variant_override = self.flags.gate_up_variant.clone();
        // (kernel_name, kernel_src, m_tile, block_threads). m_tile is the
        // per-block row count; block_threads is the wave/block size.
        let (kernel_name, kernel_src, m_tile, block_threads) = match variant_override.as_deref() {
            Some("ldsx") => (
                "gemm_gate_up_hfq4g256_wmma_ldsx",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSX_SRC,
                16,
                32,
            ),
            // k4 = 4-tile pipeline (more in-flight B loads for better BW
            // utilization). Opt-in default-off; bench-measured 2026-05-21.
            Some("k4") => (
                "gemm_gate_up_hfq4g256_wmma_k4",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_K4_SRC,
                16,
                32,
            ),
            // ldscoop = cooperative LDS weight staging for coalesced DRAM
            // loads. All 32 threads load one row's weights at a time
            // (128-byte coalesced cache lines), staged in LDS for the
            // WMMA loop. Targets the 32% peak BW seen in base kernel.
            Some("ldscoop") => (
                "gemm_gate_up_hfq4g256_wmma_ldscoop",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSCOOP_SRC,
                16,
                32,
            ),
            // 2tile = 32 rows × 16 cols per block, 2 wave32 waves.
            // Halves grid in M; both waves share the same X tile so
            // L0/L1 cache absorbs the second wave's loads cheaply.
            Some("2tile") => (
                "gemm_gate_up_hfq4g256_wmma_2tile",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_2TILE_SRC,
                32,
                64,
            ),
            _ => (
                "gemm_gate_up_hfq4g256_wmma",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_SRC,
                16,
                32,
            ),
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
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
        let row_tiles = (total_m + m_tile - 1) / m_tile;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [block_threads as u32, 1, 1],
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
    pub fn gemm_gate_up_hfq3g256_wmma(
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
        let total_m = gate_m + up_m;
        let arch_supports_mb4 = self.arch_caps.is_rdna3()
            && !self.arch_caps.is_gfx1152()
            && !self.arch_caps.is_gfx1103();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && batch_size >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_gate_up_hfq3g256_wmma_mb4(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_gate_up_hfq3g256_wmma_gfx12(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_wmma",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_SRC,
            "gemm_gate_up_hfq3g256_wmma",
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

        let weight_bytes = (gate_m + up_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma",
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
    /// HFQ3 gate_up mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_gate_up_hfq3g256_wmma_mb4(
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
            "gemm_gate_up_hfq3g256_wmma_mb4",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_MB4_SRC,
            "gemm_gate_up_hfq3g256_wmma_mb4",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m_v = gate_m as i32;
        let mut u_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m_v as *mut _ as *mut c_void,
            &mut u_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = total_m * (k / 256) * 104 + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_wmma_mb4", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma_mb4",
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
                b.push_i32(g_m_v);
                b.push_i32(u_m_v);
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
    /// MQ3 wrapper for `gemm_gate_up_hfq3g256_wmma`: pre-rotates X then
    /// dispatches the HFQ3 kernel. See `gemm_qkvza_mq3g256_wmma` for
    /// the cache-invalidation rationale.
    pub fn gemm_gate_up_mq3g256_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
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
        self.gemm_gate_up_hfq3g256_wmma(
            a_gate, a_up, x_rot, y_gate, y_up, gate_m, up_m, k, batch_size,
        )
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
    #[allow(clippy::too_many_arguments)]
    fn gemm_gate_up_moe_scalar_batched(
        &mut self,
        kernel_name: &'static str,
        weight_stride: usize,
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
            kernel_name,
            kernels::MOE_MQ_GFX1151_SCALAR_BATCHED_SRC,
            kernel_name,
        )?;
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let total_m = gate_m + up_m;
        let bytes = total_m * (k / 256) * weight_stride + batch_size * (k + total_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [total_m as u32, batch_size as u32, 1],
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
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq2g256(
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
        self.gemm_gate_up_moe_scalar_batched(
            "gemm_gate_up_hfq2g256_scalar_batched",
            72,
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq8g256(
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
        self.gemm_gate_up_moe_scalar_batched(
            "gemm_gate_up_hfq8g256_scalar_batched",
            258,
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_mq2g256_lloyd_batched(
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
        self.gemm_gate_up_moe_scalar_batched(
            "gemm_gate_up_mq2g256_lloyd_scalar_batched",
            72,
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
        )
    }
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
    /// Batched HFQ4-G256 fused 2-way gate_up GEMM with dp4a inner loop on
    /// gfx906. HFQ4 sibling of `gemm_gate_up_hfq6g256_wave64_dp4a`. Closes
    /// the dispatch fallthrough where MQ4 at gfx906 batched FFN preamble
    /// drops to `gemm_gate_up_hfq4g256_fp16_wave64`. Issue #276 Gap 2.
    pub fn gemm_gate_up_hfq4g256_wave64_dp4a(
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
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_gate_up_hfq4g256_wave64_dp4a_prequant(
            a_gate, a_up, xq_ptr, y_gate, y_up, gate_m, up_m, k, batch_size,
        )
    }
    /// Prequant entry point — see `gemm_qkvza_hfq4g256_wave64_dp4a_prequant`
    /// for rationale.
    pub fn gemm_gate_up_hfq4g256_wave64_dp4a_prequant(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        xq_ptr: *mut c_void,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_wave64_dp4a",
            kernels::GEMM_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_gate_up_hfq4g256_wave64_dp4a",
        )?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::hfq4g256_weight_bytes(gate_m, k)
            + crate::profile::hfq4g256_weight_bytes(up_m, k)
            + batch_size * k
            + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq4g256_wave64_dp4a",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq4g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xq);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
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
    /// Batched 2-way fused HFQ6-G256 GEMM for the FFN preamble (gate + up).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256(
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
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_gate_up_hfq6g256_wmma_gfx12(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_gate_up_hfq6g256_wmma(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3).
            // Skip in capture mode (Q8_1 quantize) — matches HFQ4 sibling.
            if self.arch_caps.gemv_dp4a_enabled() && !self.capture_mode {
                return self.gemm_gate_up_hfq6g256_wave64_dp4a(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq6g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_gate_up_hfq6g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256",
            kernels::GEMM_GATE_UP_HFQ6G256_SRC,
            "gemm_gate_up_hfq6g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq6g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256", bytes);
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
    /// FP16-packed batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256_fp16(
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
            "gemm_gate_up_hfq6g256_fp16",
            kernels::GEMM_GATE_UP_HFQ6G256_FP16_SRC,
            "gemm_gate_up_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq6g256_fp16"];

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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256_fp16", bytes);
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
    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 2-way fused gate+up GEMM. Phase A.3.
    pub fn gemm_gate_up_hfq6g256_wave64_dp4a(
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
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_wave64_dp4a",
            kernels::GEMM_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_gate_up_hfq6g256_wave64_dp4a",
        )?;

        let agate = a_gate.buf.as_ptr();
        let aup = a_up.buf.as_ptr();
        let ygate = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &agate as *const _ as *mut c_void,
            &aup as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &ygate as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(agate);
                b.push_ptr(aup);
                b.push_ptr(xq);
                b.push_ptr(ygate);
                b.push_ptr(yup);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }
    pub fn gemm_gate_up_hfq6g256_dot2(
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
            "gemm_gate_up_hfq6g256_dot2",
            kernels::GEMM_GATE_UP_HFQ6G256_DOT2_SRC,
            "gemm_gate_up_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq6g256_dot2"];

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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

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
    /// WMMA-accelerated batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256_wmma(
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
            "gemm_gate_up_hfq6g256_wmma",
            kernels::GEMM_GATE_UP_HFQ6G256_WMMA_SRC,
            "gemm_gate_up_hfq6g256_wmma",
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
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wmma",
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
    /// WMMA 2-way fused Q8_0 GEMM (w_gate + w_up). FFN preamble.
    /// Auto-routes to gfx12 sibling on RDNA4.
    pub fn gemm_gate_up_q8_0_wmma(
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
        if self.arch_caps.is_rdna4() {
            return self.gemm_gate_up_q8_0_wmma_gfx12(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_gate_up_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );
        static Q8_GATE_UP_4W: OnceLock<Option<bool>> = OnceLock::new();
        let q8_gate_up_4w = Self::gfx1151_q8_4w_enabled(
            *Q8_GATE_UP_4W.get_or_init(|| Self::q8_4w_mode("HIPFIRE_Q8_GATE_UP_4W")),
            batch_size >= 128,
        );
        if q8_gate_up_4w && self.arch == "gfx1151" && batch_size % 64 == 0 {
            return self.gemm_gate_up_q8_0_wmma_4w_gfx1151(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_q8_0_wmma",
            kernels::GEMM_GATE_UP_Q8_0_WMMA_SRC,
            "gemm_gate_up_q8_0_wmma",
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
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_q8_0_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_q8_0_wmma",
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
        let result = self.launch_maybe_blob(
            "gemm_gate_up_q8_0_wmma_4w_gfx1151",
            [row_tiles as u32, batch_tiles as u32, 1],
            [128, 1, 1],
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
}
