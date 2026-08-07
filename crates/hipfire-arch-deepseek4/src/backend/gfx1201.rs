// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_dispatch::families::moe::MoeBiasAwareMq2Backend;
use rdna_compute::{Gpu, GpuTensor};

/// Model-side proof that the frozen MQ2R recipe was admitted on exact gfx1201.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Gfx1201Backend {
    _sealed: (),
}

impl Gfx1201Backend {
    pub(super) fn try_new(gpu: &mut Gpu) -> Option<Self> {
        gpu.try_gfx1201().map(|_| Self { _sealed: () })
    }

    fn verify(gpu: &mut Gpu, operation: &str) -> Result<(), String> {
        gpu.try_gfx1201().map(|_| ()).ok_or_else(|| {
            format!("deepseek4: loaded gfx1201 backend cannot execute {operation} on this GPU")
        })
    }
}

impl MoeBiasAwareMq2Backend for Gfx1201Backend {
    fn gate_up(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String> {
        if let Some(dummy) = nonowned_gate_up_dummy {
            gpu.try_gfx1201()
                .ok_or_else(|| {
                    "deepseek4: loaded gfx1201 backend cannot execute MQ2 gate-up on this GPU"
                        .to_owned()
                })?
                .mq2_lloyd_moe_gate_up_ep(
                    expert_ptrs,
                    dummy,
                    topk_indices,
                    x_rot,
                    y_gate,
                    y_up,
                    m,
                    k,
                    k_top,
                )
                .map_err(|e| format!("gfx1201 MQ2 EP gate-up: {e:?}"))
        } else {
            Self::verify(gpu, "MQ2 gate-up")?;
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                expert_ptrs,
                topk_indices,
                x_rot,
                y_gate,
                y_up,
                m,
                k,
                k_top,
            )
            .map_err(|e| format!("gfx1201 portable MQ2 gate-up: {e:?}"))
        }
    }

    fn rotate_x_batched(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        Self::verify(gpu, "MQ rotation")?;
        gpu.rotate_x_mq_batched(x, x_rot, k, batch_size)
            .map_err(|e| format!("gfx1201 MQ rotate: {e:?}"))
    }

    fn down_expanded(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        if let Some(dummy) = nonowned_gate_up_dummy {
            gpu.try_gfx1201()
                .ok_or_else(|| {
                    "deepseek4: loaded gfx1201 backend cannot execute MQ2 down on this GPU"
                        .to_owned()
                })?
                .mq2_lloyd_moe_down_expanded_lds_ep(
                    expert_ptrs,
                    ownership_ptrs,
                    dummy,
                    topk_indices,
                    rot_batch,
                    expert_outputs,
                    m,
                    k,
                    k_top,
                    batch_size,
                )
                .map_err(|e| format!("gfx1201 MQ2 EP deterministic down: {e:?}"))
        } else {
            Self::verify(gpu, "MQ2 down")?;
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                expert_ptrs,
                topk_indices,
                rot_batch,
                expert_outputs,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| format!("gfx1201 portable MQ2 deterministic down: {e:?}"))
        }
    }

    fn down_residual_scaled(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String> {
        Self::verify(gpu, "MQ2 hash down")?;
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
            expert_ptrs,
            topk_indices,
            topk_weights,
            rot_batch,
            residual,
            m,
            k,
            k_top,
            false,
        )
        .map_err(|e| format!("gfx1201 MQ2 hash down: {e:?}"))
    }
}
