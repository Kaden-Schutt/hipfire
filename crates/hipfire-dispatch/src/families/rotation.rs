// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::types::*;
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// Parameters for a rotation family dispatch call.
pub struct RotationParams<'a> {
    pub x: &'a GpuTensor,
    pub x_up: Option<&'a GpuTensor>,
    pub w_norm: Option<&'a GpuTensor>,
    pub x_plain: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub awq_scale: Option<&'a GpuTensor>,
    pub k: usize,
    pub eps: f32,
    pub batch_size: usize,
    pub variant: RotationVariant,
}

/// Rotation kernel family — selects and runs FWHT rotation kernels.
pub struct RotationFamily {
    registry: KernelRegistry,
}

impl RotationFamily {
    pub fn new() -> Self {
        let registry = KernelRegistry::new();
        super::super::tables::rotation_table::populate(&registry);
        Self { registry }
    }

    /// Run the selected rotation kernel.
    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: RotationParams<'_>,
    ) -> Result<(), hip_bridge::HipError> {
        use hip_bridge::HipError;
        let he = |e: crate::types::DispatchError| HipError::new(0, &e.to_string());

        let has_awq = params.awq_scale.is_some();
        let batched = params.batch_size > 1;

        match params.variant {
            RotationVariant::PlainG128 => {
                self.registry.resolve(KernelKey::RotateMqG128, ctx)
                    .map_err(he)?;
                // rotate_x_mq_128 internally calls ensure_mq_signs_128()
                gpu.rotate_x_mq_128(params.x, params.x_rot, params.k)
            }
            RotationVariant::Plain => match (has_awq, batched) {
                (false, false) => {
                    self.registry.resolve(KernelKey::RotateMq, ctx)
                        .map_err(he)?;
                    gpu.rotate_x_mq(params.x, params.x_rot, params.k)
                }
                (true, false) => {
                    self.registry.resolve(KernelKey::RotateMqAwq, ctx)
                        .map_err(he)?;
                    gpu.rotate_x_mq_awq(
                        params.x,
                        params.awq_scale.unwrap(),
                        params.x_rot,
                        params.k,
                    )
                }
                (false, true) => {
                    self.registry.resolve(KernelKey::RotateMqBatched, ctx)
                        .map_err(he)?;
                    gpu.rotate_x_mq(params.x, params.x_rot, params.k)
                }
                (true, true) => {
                    self.registry.resolve(KernelKey::RotateMqAwqBatched, ctx)
                        .map_err(he)?;
                    gpu.rotate_x_mq_awq(
                        params.x,
                        params.awq_scale.unwrap(),
                        params.x_rot,
                        params.k,
                    )
                }
            },
            RotationVariant::WithRmsnorm => {
                let w_norm = params.w_norm.ok_or_else(|| {
                    HipError::new(0, "w_norm required for WithRmsnorm rotation")
                })?;
                match (has_awq, batched) {
                    (false, false) => {
                        self.registry.resolve(KernelKey::RmsnormRotateMq, ctx)
                            .map_err(he)?;
                        gpu.fused_rmsnorm_rotate_mq(
                            params.x,
                            w_norm,
                            params.x_rot,
                            params.k,
                            params.eps,
                        )
                    }
                    (true, false) => {
                        self.registry.resolve(KernelKey::RmsnormRotateMqAwq, ctx)
                            .map_err(he)?;
                        gpu.fused_rmsnorm_rotate_mq_awq(
                            params.x,
                            w_norm,
                            params.awq_scale.unwrap(),
                            params.x_rot,
                            params.k,
                            params.eps,
                        )
                    }
                    (false, true) => {
                        self.registry.resolve(KernelKey::RmsnormRotateMqBatched, ctx)
                            .map_err(he)?;
                        gpu.fused_rmsnorm_rotate_mq_batched(
                            params.x,
                            w_norm,
                            params.x_rot,
                            params.k,
                            params.eps,
                            params.batch_size,
                        )
                    }
                    (true, true) => {
                        self.registry
                            .resolve(KernelKey::RmsnormRotateMqAwqBatched, ctx)
                            .map_err(he)?;
                        gpu.fused_rmsnorm_rotate_mq_awq_batched(
                            params.x,
                            w_norm,
                            params.awq_scale.unwrap(),
                            params.x_rot,
                            params.k,
                            params.eps,
                            params.batch_size,
                        )
                    }
                }
            }
            RotationVariant::WithSwiGLU => {
                let x_up = params.x_up.ok_or_else(|| {
                    HipError::new(0, "x_up required for WithSwiGLU rotation")
                })?;
                match has_awq {
                    false => {
                        self.registry.resolve(KernelKey::SiluMulRotateMq, ctx)
                            .map_err(he)?;
                        gpu.fused_silu_mul_rotate_mq(
                            params.x,
                            x_up,
                            params.x_rot,
                            params.k,
                        )
                    }
                    true => {
                        self.registry.resolve(KernelKey::SiluMulRotateMqAwq, ctx)
                            .map_err(he)?;
                        gpu.fused_silu_mul_rotate_mq_awq(
                            params.x,
                            x_up,
                            params.awq_scale.unwrap(),
                            params.x_rot,
                            params.k,
                        )
                    }
                }
            }
        }
    }
}
