// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RMSNorm dispatch (f32, batched, train fwd/bwd, slot-buf). Pure move (Phase 1 M1).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// out = rmsnorm(x, weight, eps)
    pub fn rmsnorm_f32(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (module_name, kernel_src, kernel_name, timer_name, reduce_slots) =
            if self.arch_caps.is_gfx1151() {
                (
                    "rmsnorm_gfx1151",
                    kernels::RMSNORM_GFX1151_SRC,
                    "rmsnorm_f32_gfx1151",
                    "rmsnorm_f32_gfx1151",
                    8u32,
                )
            } else {
                (
                    "rmsnorm",
                    kernels::RMSNORM_SRC,
                    "rmsnorm_f32",
                    "rmsnorm_f32",
                    256u32,
                )
            };
        self.ensure_kernel(module_name, kernel_src, kernel_name)?;

        let batch = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let x_ptr = x.buf.as_ptr();
        let w_ptr = weight.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_val = n;
        let eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &eps_val as *const _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = if self.arch_caps.is_gfx1151() {
            reduce_slots * 4
        } else {
            block_size * 4
        };

        let bytes = crate::profile::rmsnorm_bytes(batch * n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", timer_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [batch as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(w_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b.push_f32(eps_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn rmsnorm_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        batch: usize,
        n: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (module_name, kernel_src, kernel_name, timer_name, reduce_slots) =
            if self.arch_caps.is_gfx1151() {
                (
                    "rmsnorm_gfx1151",
                    kernels::RMSNORM_GFX1151_SRC,
                    "rmsnorm_f32_gfx1151",
                    "rmsnorm_batched_gfx1151",
                    8u32,
                )
            } else {
                (
                    "rmsnorm",
                    kernels::RMSNORM_SRC,
                    "rmsnorm_f32",
                    "rmsnorm_batched",
                    256u32,
                )
            };
        self.ensure_kernel(module_name, kernel_src, kernel_name)?;

        let mut x_ptr = x.buf.as_ptr();
        let mut w_ptr = weight.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n as i32;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = if self.arch_caps.is_gfx1151() {
            reduce_slots * 4
        } else {
            block_size * 4
        };
        let bytes = crate::profile::rmsnorm_bytes(batch * n);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", timer_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [batch as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(w_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b.push_f32(eps_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Training RMSNorm forward (fp32). `x`,`y`: `[rows*H]`; `w`: `[H]`;
    /// `rinv`: `[rows]` output (1/r per row, consumed by the backward).
    pub fn rmsnorm_train_fwd(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        rinv: &GpuTensor,
        rows: usize,
        h: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rmsnorm_train_fwd",
            kernels::RMSNORM_TRAIN_SRC,
            "rmsnorm_train_fwd",
        )?;
        let func = &self.functions["rmsnorm_train_fwd"];
        let mut xp = x.buf.as_ptr();
        let mut wp = w.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut rp = rinv.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut hi = h as i32;
        let mut epsf = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut epsf as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Training RMSNorm backward (fp32). Produces `dx` `[rows*H]` and
    /// atomic-accumulates `dw` `[H]` (zero it first). `rinv` is from the forward.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_train_bwd(
        &mut self,
        dy: &GpuTensor,
        x: &GpuTensor,
        w: &GpuTensor,
        rinv: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        rows: usize,
        h: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rmsnorm_train_bwd",
            kernels::RMSNORM_TRAIN_SRC,
            "rmsnorm_train_bwd",
        )?;
        let func = &self.functions["rmsnorm_train_bwd"];
        let mut dyp = dy.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut wp = w.buf.as_ptr();
        let mut rp = rinv.buf.as_ptr();
        let mut dxp = dx.buf.as_ptr();
        let mut dwp = dw.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut hi = h as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dyp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut dxp as *mut _ as *mut c_void,
            &mut dwp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HIP-graphs-safe in-place RMSNorm at `base + slot_buf[0] * n`.
    /// -1 sentinel → no-op. Single block (head_dim ≤ 512).
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn rmsnorm_f32_at_slot_buf(
        &mut self,
        base: &GpuTensor,
        weight: &GpuTensor,
        slot_buf: &GpuTensor,
        n: i32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rmsnorm_f32_at_slot_buf",
            kernels::RMSNORM_AT_SLOT_BUF_SRC,
            "rmsnorm_f32_at_slot_buf",
        )?;
        let bp = base.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let mut nv = n;
        let mut ev = eps;
        let mut params: Vec<*mut c_void> = vec![
            &bp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut ev as *mut _ as *mut c_void,
        ];
        let block = 256u32.min(n as u32).next_power_of_two().max(32);
        let shared = block * 4;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(bp);
            b.push_ptr(wp);
            b.push_ptr(sb);
            b.push_i32(nv);
            b.push_f32(ev);
            b
        };
        self.launch_maybe_blob(
            "rmsnorm_f32_at_slot_buf",
            [1, 1, 1],
            [block, 1, 1],
            shared,
            &mut params,
            blob_builder,
        )
    }
}
