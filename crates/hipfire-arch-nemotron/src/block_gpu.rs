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
use crate::weight::LinearWeight;
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

const F32: usize = std::mem::size_of::<f32>();

/// GPU-resident weights + recurrent state + scratch for one Mamba-2 block.
pub struct Mamba2BlockGpu {
    pub dims: Mamba2Dims,
    // weights — in_proj/out_proj are F32 or quantized (LinearWeight); the
    // recurrence tensors (conv/A_log/D/dt_bias/norm) are always F32.
    in_proj: LinearWeight,
    conv_weight: GpuTensor,
    conv_bias: GpuTensor,
    a_log: GpuTensor,
    d: GpuTensor,
    dt_bias: GpuTensor,
    norm_weight: GpuTensor,
    out_proj: LinearWeight,
    /// Post-`out_proj` scalar (nemotron_h residual rescale `1/√num_layers`).
    /// The f32 path folds this into the weight at load (so it's `1.0` here); the
    /// HFQ path can't rescale quantized bytes, so it carries the scale and
    /// applies it to the gemv output instead. Both yield the same result.
    out_proj_scale: f32,
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
        let in_proj = LinearWeight::F32(
            gpu.upload_f32(w.in_proj, &[dims.projection_size(), dims.hidden_size])?,
        );
        let out_proj =
            LinearWeight::F32(gpu.upload_f32(w.out_proj, &[dims.hidden_size, dims.d_inner()])?);
        // f32 path: out_proj already carries the residual rescale (folded in by
        // the loader), so the runtime scale is the identity.
        Self::assemble(
            gpu,
            dims,
            in_proj,
            out_proj,
            1.0,
            w.conv_weight,
            w.conv_bias,
            w.a_log,
            w.d,
            w.dt_bias,
            w.norm_weight,
        )
    }

    /// HFQ path: `in_proj`/`out_proj` are pre-built quantized [`LinearWeight`]s;
    /// the recurrence tensors come as host f32 slices (kept F16/F32 in the HFQ).
    #[allow(clippy::too_many_arguments)]
    pub fn new_quant(
        gpu: &mut Gpu,
        dims: Mamba2Dims,
        in_proj: LinearWeight,
        out_proj: LinearWeight,
        out_proj_scale: f32,
        conv_weight: &[f32],
        conv_bias: &[f32],
        a_log: &[f32],
        d: &[f32],
        dt_bias: &[f32],
        norm_weight: &[f32],
    ) -> HipResult<Self> {
        Self::assemble(
            gpu,
            dims,
            in_proj,
            out_proj,
            out_proj_scale,
            conv_weight,
            conv_bias,
            a_log,
            d,
            dt_bias,
            norm_weight,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        gpu: &mut Gpu,
        dims: Mamba2Dims,
        in_proj: LinearWeight,
        out_proj: LinearWeight,
        out_proj_scale: f32,
        conv_weight: &[f32],
        conv_bias: &[f32],
        a_log: &[f32],
        d: &[f32],
        dt_bias: &[f32],
        norm_weight: &[f32],
    ) -> HipResult<Self> {
        let d_inner = dims.d_inner();
        let conv_dim = dims.conv_dim();
        let nh = dims.num_heads;
        let nss = dims.n_groups * dims.state_size;
        let proj_sz = dims.projection_size();
        let hist = dims.conv_kernel - 1;

        Ok(Self {
            in_proj,
            out_proj,
            out_proj_scale,
            conv_weight: gpu.upload_f32(conv_weight, &[conv_dim, dims.conv_kernel])?,
            conv_bias: gpu.upload_f32(conv_bias, &[conv_dim])?,
            a_log: gpu.upload_f32(a_log, &[nh])?,
            d: gpu.upload_f32(d, &[nh])?,
            dt_bias: gpu.upload_f32(dt_bias, &[nh])?,
            norm_weight: gpu.upload_f32(norm_weight, &[d_inner])?,
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
        self.in_proj.gemv(gpu, hidden, &self.proj)?;

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

        // 7. out_proj (+ residual rescale on the HFQ path; identity on f32)
        self.out_proj.gemv(gpu, &self.y_norm, &self.out)?;
        if self.out_proj_scale != 1.0 {
            gpu.scale_f32(&self.out, self.out_proj_scale)?;
        }
        Ok(&self.out)
    }

    /// Zero the recurrent conv + SSM state for a fresh sequence.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        gpu.fill_f32(&self.conv_state, 0.0)?;
        gpu.fill_f32(&self.ssm_state, 0.0)?;
        Ok(())
    }

    /// Free all GPU tensors (consumes the block).
    pub fn free(self, gpu: &mut Gpu) {
        self.in_proj.free(gpu);
        self.out_proj.free(gpu);
        for t in [
            self.conv_weight,
            self.conv_bias,
            self.a_log,
            self.d,
            self.dt_bias,
            self.norm_weight,
            self.conv_state,
            self.ssm_state,
            self.proj,
            self.xbc_in,
            self.xbc_act,
            self.dt_raw,
            self.z,
            self.x,
            self.b,
            self.c,
            self.y,
            self.y_norm,
            self.out,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}
