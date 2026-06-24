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

    /// Mamba-2 SSD **prefill** scan (N6): process a whole `seq_len`-token prompt
    /// in ONE launch instead of `seq_len` `mamba2_ssd_decode_f32` launches,
    /// updating `state` in place to the post-sequence value (decode hand-off).
    /// Bit-faithful to the sequential decode (`ssd::ssd_sequence`). See
    /// `kernels/src/mamba2_ssd_seq.hip`.
    ///
    /// - `y` (out): `[seq_len * num_heads * head_dim]`
    /// - `state`: `[num_heads * head_dim * state_size]` (in/out)
    /// - `x`: `[seq_len * num_heads * head_dim]`
    /// - `b`, `c`: `[seq_len * n_groups * state_size]`
    /// - `dt_raw`: `[seq_len * num_heads]`
    /// - `a_log`, `d`, `dt_bias`: `[num_heads]`
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_ssd_seq_f32(
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
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        state_size: usize,
        n_groups: usize,
        dt_min: f32,
        dt_max: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "mamba2_ssd_seq",
            kernels::MAMBA2_SSD_SEQ_SRC,
            "mamba2_ssd_seq_f32",
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
        let sl = seq_len as i32;
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
            &sl as *const _ as *mut c_void,
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
        let func = &self.functions["mamba2_ssd_seq_f32"];
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

    /// Mamba-2 `RMSNormGated` (gate-then-group-RMSNorm): `out = group_rmsnorm(y *
    /// silu(z)) * weight`, over `n_groups` groups of `group_size`
    /// (`group_size = d_inner / n_groups`). See
    /// `kernels/src/mamba2_gated_norm.hip`. (Distinct from `gated_norm_f32`,
    /// which is qwen35's norm-then-gate per-head.)
    pub fn mamba2_gated_norm_f32(
        &mut self,
        out: &GpuTensor,
        y: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        n_groups: usize,
        group_size: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "mamba2_gated_norm",
            kernels::MAMBA2_GATED_NORM_SRC,
            "mamba2_gated_norm_f32",
        )?;
        let op = out.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let ng = n_groups as i32;
        let gs = group_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ng as *const _ as *mut c_void,
            &gs as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        let func = &self.functions["mamba2_gated_norm_f32"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_groups as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Mamba-2 `RMSNormGated` **prefill** form (N6): batched gate-then-group-
    /// RMSNorm over `seq_len` positions in one launch (vs `seq_len`
    /// `mamba2_gated_norm_f32` launches). `y`/`z`/`out` are `[seq_len * d_inner]`
    /// (position-major); `weight` is `[d_inner]` (shared across positions). See
    /// `kernels/src/mamba2_gated_norm_seq.hip`.
    pub fn mamba2_gated_norm_seq_f32(
        &mut self,
        out: &GpuTensor,
        y: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        seq_len: usize,
        n_groups: usize,
        group_size: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "mamba2_gated_norm_seq",
            kernels::MAMBA2_GATED_NORM_SEQ_SRC,
            "mamba2_gated_norm_seq_f32",
        )?;
        let op = out.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sl = seq_len as i32;
        let ng = n_groups as i32;
        let gs = group_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sl as *const _ as *mut c_void,
            &ng as *const _ as *mut c_void,
            &gs as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        let func = &self.functions["mamba2_gated_norm_seq_f32"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_groups as u32, seq_len as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
