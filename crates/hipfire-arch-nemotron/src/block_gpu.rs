// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mamba-2 mixer **block** decode step — GPU forward (N3).
//!
//! Chains the four validated HIP kernels in the order the CPU oracle
//! ([`crate::block::mamba2_block_decode_step`]) defines, slicing the in_proj /
//! conv outputs on-device with `memcpy_dtod`:
//! ```text
//!   proj = gemv_f32(in_proj, hidden)              # [projection_size]
//!   z, xBC, dt = dtod-slice(proj)                 # gate | conv-in | dt
//!   xBC = conv1d_bias_silu_decode_f32(xBC)        # depthwise K=4 + bias + SiLU
//!   x, B, C = dtod-slice(xBC)                      # d_inner | n_groups*ssm | ..
//!   y = mamba2_ssd_decode_f32(x, B, C, dt)        # selective scan
//!   y = mamba2_gated_norm_f32(y, z)               # gate-then-group-RMSNorm
//!   out = gemv_f32(out_proj, y)                    # [hidden_size]
//! ```
//! Validated gpu-vs-cpu against the oracle in
//! `examples/test_block_gpu.rs`. f32 / decode-only; chunked prefill is N6.

use crate::block::{Mamba2BlockWeights, Mamba2Dims};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

const F32: usize = std::mem::size_of::<f32>();

/// GPU-resident weights + recurrent state + scratch for one Mamba-2 block.
pub struct Mamba2BlockGpu {
    pub dims: Mamba2Dims,
    // weights
    in_proj: GpuTensor,
    conv_weight: GpuTensor,
    conv_bias: GpuTensor,
    a_log: GpuTensor,
    d: GpuTensor,
    dt_bias: GpuTensor,
    norm_weight: GpuTensor,
    out_proj: GpuTensor,
    // recurrent state (zero-initialized)
    conv_state: GpuTensor,
    ssm_state: GpuTensor,
    // scratch (reused across steps)
    proj: GpuTensor,
    xbc_in: GpuTensor,
    xbc_act: GpuTensor,
    dt_raw: GpuTensor,
    z: GpuTensor,
    x: GpuTensor,
    b: GpuTensor,
    c: GpuTensor,
    y: GpuTensor,
    y_norm: GpuTensor,
    out: GpuTensor,
}

impl Mamba2BlockGpu {
    /// Upload `w` (host f32 slices) and allocate state + scratch. State starts
    /// at zero (matching the CPU oracle's `Mamba2BlockState::zeros`).
    pub fn new(gpu: &mut Gpu, dims: Mamba2Dims, w: &Mamba2BlockWeights) -> HipResult<Self> {
        let d_inner = dims.d_inner();
        let conv_dim = dims.conv_dim();
        let nh = dims.num_heads;
        let nss = dims.n_groups * dims.state_size;
        let proj_sz = dims.projection_size();
        let hist = dims.conv_kernel - 1;

        Ok(Self {
            in_proj: gpu.upload_f32(w.in_proj, &[proj_sz, dims.hidden_size])?,
            conv_weight: gpu.upload_f32(w.conv_weight, &[conv_dim, dims.conv_kernel])?,
            conv_bias: gpu.upload_f32(w.conv_bias, &[conv_dim])?,
            a_log: gpu.upload_f32(w.a_log, &[nh])?,
            d: gpu.upload_f32(w.d, &[nh])?,
            dt_bias: gpu.upload_f32(w.dt_bias, &[nh])?,
            norm_weight: gpu.upload_f32(w.norm_weight, &[d_inner])?,
            out_proj: gpu.upload_f32(w.out_proj, &[dims.hidden_size, d_inner])?,
            conv_state: gpu.zeros(&[conv_dim, hist], DType::F32)?,
            ssm_state: gpu.zeros(&[nh * dims.head_dim, dims.state_size], DType::F32)?,
            proj: gpu.zeros(&[proj_sz], DType::F32)?,
            xbc_in: gpu.zeros(&[conv_dim], DType::F32)?,
            xbc_act: gpu.zeros(&[conv_dim], DType::F32)?,
            dt_raw: gpu.zeros(&[nh], DType::F32)?,
            z: gpu.zeros(&[d_inner], DType::F32)?,
            x: gpu.zeros(&[d_inner], DType::F32)?,
            b: gpu.zeros(&[nss], DType::F32)?,
            c: gpu.zeros(&[nss], DType::F32)?,
            y: gpu.zeros(&[d_inner], DType::F32)?,
            y_norm: gpu.zeros(&[d_inner], DType::F32)?,
            out: gpu.zeros(&[dims.hidden_size], DType::F32)?,
            dims,
        })
    }

    /// One decode step. Reads `hidden` `[hidden_size]`, updates conv+ssm state in
    /// place, returns the `[hidden_size]` mixer output tensor.
    pub fn decode_step(&mut self, gpu: &mut Gpu, hidden: &GpuTensor) -> HipResult<&GpuTensor> {
        let d = &self.dims;
        let d_inner = d.d_inner();
        let conv_dim = d.conv_dim();
        let nss = d.n_groups * d.state_size;

        // 1. in_proj
        gpu.gemv_f32(&self.in_proj, hidden, &self.proj)?;

        // 2. slice proj → z | xBC | dt_raw
        gpu.memcpy_dtod_at_auto(&self.z.buf, 0, &self.proj.buf, 0, d_inner * F32)?;
        gpu.memcpy_dtod_at_auto(
            &self.xbc_in.buf,
            0,
            &self.proj.buf,
            d_inner * F32,
            conv_dim * F32,
        )?;
        gpu.memcpy_dtod_at_auto(
            &self.dt_raw.buf,
            0,
            &self.proj.buf,
            (d_inner + conv_dim) * F32,
            d.num_heads * F32,
        )?;

        // 3. conv1d + bias + SiLU
        gpu.conv1d_bias_silu_decode_f32(
            &self.xbc_act,
            &self.xbc_in,
            &self.conv_weight,
            &self.conv_bias,
            &self.conv_state,
            conv_dim,
        )?;

        // 4. slice xBC_act → x | B | C
        gpu.memcpy_dtod_at_auto(&self.x.buf, 0, &self.xbc_act.buf, 0, d_inner * F32)?;
        gpu.memcpy_dtod_at_auto(&self.b.buf, 0, &self.xbc_act.buf, d_inner * F32, nss * F32)?;
        gpu.memcpy_dtod_at_auto(
            &self.c.buf,
            0,
            &self.xbc_act.buf,
            (d_inner + nss) * F32,
            nss * F32,
        )?;

        // 5. SSD selective scan
        gpu.mamba2_ssd_decode_f32(
            &self.y,
            &self.ssm_state,
            &self.x,
            &self.b,
            &self.c,
            &self.dt_raw,
            &self.a_log,
            &self.d,
            &self.dt_bias,
            d.num_heads,
            d.head_dim,
            d.state_size,
            d.n_groups,
            d.dt_min,
            d.dt_max,
        )?;

        // 6. RMSNormGated (num norm groups = d_inner / group_size = n_groups)
        let group_size = d.norm_group_size();
        let num_norm_groups = d_inner / group_size;
        gpu.mamba2_gated_norm_f32(
            &self.y_norm,
            &self.y,
            &self.z,
            &self.norm_weight,
            num_norm_groups,
            group_size,
            d.rms_norm_eps,
        )?;

        // 7. out_proj
        gpu.gemv_f32(&self.out_proj, &self.y_norm, &self.out)?;
        Ok(&self.out)
    }
}
