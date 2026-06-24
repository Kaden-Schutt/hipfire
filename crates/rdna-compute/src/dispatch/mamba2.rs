// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mamba-2 SSD (selective state-space) dispatch — the nemotron_h Mamba-2 mixer
//! recurrence. N1: single-token f32 decode (`mamba2_ssd_decode_f32`), the
//! reference floor validated gpu-vs-cpu against
//! `hipfire_arch_nemotron::ssd::ssd_decode_step`. Chunked-SSD prefill + q8 state
//! land later.

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// One Mamba-2 SSD decode step (single token), updating `state` in place and
    /// writing the mixer output `y`. See `kernels/src/mamba2_ssd_decode.hip`.
    ///
    /// - `y`, `x`: `[num_heads * head_dim]`
    /// - `state`: `[num_heads * head_dim * state_size]` (updated in place)
    /// - `b`, `c`: `[n_groups * state_size]`
    /// - `dt_raw`, `a_log`, `d`, `dt_bias`: `[num_heads]`
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_ssd_decode_f32(
        &mut self,
        y: &GpuTensor,
        state: &GpuTensor,
        x: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        dt_raw: &GpuTensor,
        a_log: &GpuTensor,
        d: &GpuTensor,
        dt_bias: &GpuTensor,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        n_groups: usize,
        dt_min: f32,
        dt_max: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "mamba2_ssd_decode",
            kernels::MAMBA2_SSD_DECODE_SRC,
            "mamba2_ssd_decode_f32",
        )?;
        let yp = y.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let bp = b.buf.as_ptr();
        let cp = c.buf.as_ptr();
        let dtp = dt_raw.buf.as_ptr();
        let ap = a_log.buf.as_ptr();
        let dp = d.buf.as_ptr();
        let dbp = dt_bias.buf.as_ptr();
        let nh = num_heads as i32;
        let hd = head_dim as i32;
        let ns = state_size as i32;
        let ng = n_groups as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &dtp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &dbp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ns as *const _ as *mut c_void,
            &ng as *const _ as *mut c_void,
            &dt_min as *const _ as *mut c_void,
            &dt_max as *const _ as *mut c_void,
        ];
        let total = (num_heads * head_dim) as u32;
        let block = 256u32;
        let grid = total.div_ceil(block);
        let func = &self.functions["mamba2_ssd_decode_f32"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
