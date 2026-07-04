// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! nemotron_h dense MLP (`-`) block — ReLU² FFN, CPU oracle + GPU forward.
//!
//! `out = down_proj @ relu2(up_proj @ x)` — a plain (un-gated) MLP with a
//! squared-ReLU activation (`mlp_hidden_act = "relu2"`), NOT SwiGLU. Both
//! projections are bias-free (`mlp_bias = false`). Shapes:
//! `up_proj [intermediate, hidden]`, `down_proj [hidden, intermediate]`.

use crate::weight::LinearWeight;
use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};

#[inline]
fn relu2(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

/// Row-major matvec `out[i] = Σ_j w[i*in + j] * x[j]`, `w` is `[out, in]`.
fn matvec(w: &[f32], x: &[f32], out: usize, n_in: usize, dst: &mut [f32]) {
    for i in 0..out {
        let row = &w[i * n_in..i * n_in + n_in];
        dst[i] = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// CPU reference: `down @ relu2(up @ x)`. `up` is `[intermediate, hidden]`,
/// `down` is `[hidden, intermediate]`; returns `[hidden]`.
pub fn mlp_relu2(
    up: &[f32],
    down: &[f32],
    x: &[f32],
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    let mut u = vec![0.0f32; intermediate];
    matvec(up, x, intermediate, hidden, &mut u);
    for v in u.iter_mut() {
        *v = relu2(*v);
    }
    let mut out = vec![0.0f32; hidden];
    matvec(down, &u, hidden, intermediate, &mut out);
    out
}

/// GPU-resident ReLU² MLP block (`up`/`down` weights + reused scratch).
pub struct MlpRelu2Gpu {
    hidden: usize,
    intermediate: usize,
    up: LinearWeight,
    down: LinearWeight,
    u: GpuTensor,
    a: GpuTensor,
    out: GpuTensor,
}

impl MlpRelu2Gpu {
    /// (weight-buf ptr, `mixer.`-relative name) per dense projection — calibration
    /// capture-name map. See `calibration::build_capture_names`.
    pub(crate) fn calib_projections(&self) -> [(usize, &'static str); 2] {
        [
            (self.up.buf_ptr(), "up_proj"),
            (self.down.buf_ptr(), "down_proj"),
        ]
    }

    pub fn new(
        gpu: &mut Gpu,
        hidden: usize,
        intermediate: usize,
        up: &[f32],
        down: &[f32],
    ) -> HipResult<Self> {
        let up = LinearWeight::F32(gpu.upload_f32(up, &[intermediate, hidden])?);
        let down = LinearWeight::F32(gpu.upload_f32(down, &[hidden, intermediate])?);
        Self::assemble(gpu, hidden, intermediate, up, down)
    }

    /// HFQ path: `up`/`down` are pre-built quantized [`LinearWeight`]s.
    pub fn new_quant(
        gpu: &mut Gpu,
        hidden: usize,
        intermediate: usize,
        up: LinearWeight,
        down: LinearWeight,
    ) -> HipResult<Self> {
        Self::assemble(gpu, hidden, intermediate, up, down)
    }

    fn assemble(
        gpu: &mut Gpu,
        hidden: usize,
        intermediate: usize,
        up: LinearWeight,
        down: LinearWeight,
    ) -> HipResult<Self> {
        Ok(Self {
            hidden,
            intermediate,
            up,
            down,
            u: gpu.zeros(&[intermediate], DType::F32)?,
            a: gpu.zeros(&[intermediate], DType::F32)?,
            out: gpu.zeros(&[hidden], DType::F32)?,
        })
    }

    /// `out = down @ relu2(up @ x)`. Reads `x` `[hidden]`, returns `[hidden]`.
    pub fn forward(&mut self, gpu: &mut Gpu, x: &GpuTensor) -> HipResult<&GpuTensor> {
        self.up.gemv(gpu, x, &self.u)?;
        gpu.relu2_f32(&self.u, &self.a)?;
        self.down.gemv(gpu, &self.a, &self.out)?;
        Ok(&self.out)
    }

    /// Batched prefill: `out[seq, hidden] = down @ relu2(up @ x)` over a whole
    /// prompt. MLP is position-independent, so it equals `forward` per position;
    /// `relu2_f32` is elementwise (works on `[seq*intermediate]`). Returns the
    /// `[seq * hidden]` output; scratch is allocated per call. F32 and supported
    /// HFQ/MQ/Q8 weights route through [`crate::weight::LinearWeight::gemm_seq`].
    pub fn prefill(&mut self, gpu: &mut Gpu, x: &GpuTensor, seq: usize) -> HipResult<GpuTensor> {
        let u = gpu.zeros(&[seq * self.intermediate], DType::F32)?;
        let a = gpu.zeros(&[seq * self.intermediate], DType::F32)?;
        let out = gpu.zeros(&[seq * self.hidden], DType::F32)?;
        self.up
            .gemm_seq(gpu, x, &u, seq, self.intermediate, self.hidden)?;
        gpu.relu2_f32(&u, &a)?;
        self.down
            .gemm_seq(gpu, &a, &out, seq, self.hidden, self.intermediate)?;
        let _ = gpu.free_tensor(u);
        let _ = gpu.free_tensor(a);
        Ok(out)
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn intermediate(&self) -> usize {
        self.intermediate
    }

    /// Free all GPU tensors (consumes the block).
    pub fn free(self, gpu: &mut Gpu) {
        self.up.free(gpu);
        self.down.free(gpu);
        for t in [self.u, self.a, self.out] {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_relu2_mlp_basic() {
        // hidden=2, intermediate=2, identity-ish: up=I, down=I → out=relu2(x).
        let up = vec![1.0, 0.0, 0.0, 1.0];
        let down = vec![1.0, 0.0, 0.0, 1.0];
        let out = mlp_relu2(&up, &down, &[2.0, -3.0], 2, 2);
        assert_eq!(out, vec![4.0, 0.0]); // relu2(2)=4, relu2(-3)=0
    }
}
