// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::types::*;
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
    // Givens rotation metadata (ParoQuant)
    pub givens_pairs: Option<&'a GpuTensor>,
    pub givens_theta: Option<&'a GpuTensor>,
    pub givens_scales: Option<&'a GpuTensor>,
    pub givens_krot: Option<usize>,
}

/// Rotation kernel family — selects and runs FWHT rotation kernels.
pub struct RotationFamily {
    registry: KernelRegistry,
}

impl RotationFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        super::super::tables::rotation_table::populate(&mut registry);
        registry.validate().expect("rotation kernel table has empty entries");
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
            RotationVariant::Givens => {
                let pairs = params.givens_pairs.ok_or_else(|| {
                    HipError::new(0, "givens_pairs required for Givens rotation")
                })?;
                let theta = params.givens_theta.ok_or_else(|| {
                    HipError::new(0, "givens_theta required for Givens rotation")
                })?;
                let scales = params.givens_scales.ok_or_else(|| {
                    HipError::new(0, "givens_scales required for Givens rotation")
                })?;
                let krot = params.givens_krot.ok_or_else(|| {
                    HipError::new(0, "givens_krot required for Givens rotation")
                })?;
                // givens_rotate_to does copy_d2d + rotate in one kernel
                gpu.givens_rotate_to(
                    params.x, params.x_rot,
                    pairs, theta, scales,
                    1, /* seq_len */
                    params.k,
                    krot,
                )
            }
            RotationVariant::PlainG128 => {
                self.registry.resolve(KernelKey::RotateMqG128, ctx, None)
                    .map_err(he)?;
                // rotate_x_mq_128 internally calls ensure_mq_signs_128()
                gpu.rotate_x_mq_128(params.x, params.x_rot, params.k)
            }
            RotationVariant::Plain => match (has_awq, batched) {
                (false, false) => {
                    self.registry.resolve(KernelKey::RotateMq, ctx, None)
                        .map_err(he)?;
                    gpu.rotate_x_mq(params.x, params.x_rot, params.k)
                }
                (true, false) => {
                    self.registry.resolve(KernelKey::RotateMqAwq, ctx, None)
                        .map_err(he)?;
                    gpu.rotate_x_mq_awq(
                        params.x,
                        params.awq_scale.unwrap(),
                        params.x_rot,
                        params.k,
                    )
                }
                (false, true) => {
                    self.registry.resolve(KernelKey::RotateMqBatched, ctx, None)
                        .map_err(he)?;
                    gpu.rotate_x_mq(params.x, params.x_rot, params.k)
                }
                (true, true) => {
                    self.registry.resolve(KernelKey::RotateMqAwqBatched, ctx, None)
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
                        self.registry.resolve(KernelKey::RmsnormRotateMq, ctx, None)
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
                        self.registry.resolve(KernelKey::RmsnormRotateMqAwq, ctx, None)
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
                        self.registry.resolve(KernelKey::RmsnormRotateMqBatched, ctx, None)
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
                            .resolve(KernelKey::RmsnormRotateMqAwqBatched, ctx, None)
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
                        self.registry.resolve(KernelKey::SiluMulRotateMq, ctx, None)
                            .map_err(he)?;
                        gpu.fused_silu_mul_rotate_mq(
                            params.x,
                            x_up,
                            params.x_rot,
                            params.k,
                        )
                    }
                    true => {
                        self.registry.resolve(KernelKey::SiluMulRotateMqAwq, ctx, None)
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
