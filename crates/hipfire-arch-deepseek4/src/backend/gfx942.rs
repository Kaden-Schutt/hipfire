// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_dispatch::families::moe::MoeBiasAwareMq2Backend;
use rdna_compute::{Gpu, GpuTensor};

/// Model-side proof that the frozen MQ2R recipe was admitted on exact gfx942.
///
/// The field and constructor are private. Every operation reacquires the
/// rdna-compute Gfx942Device borrow, so moving the weights to another device
/// fails closed instead of silently choosing an RDNA or portable kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Gfx942Backend {
    _sealed: (),
}

impl Gfx942Backend {
    pub(super) fn try_new(gpu: &mut Gpu) -> Option<Self> {
        gpu.try_gfx942().map(|_| Self { _sealed: () })
    }

    pub(super) fn grouped_olora_e8(
        self,
        gpu: &mut Gpu,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        gpu.try_gfx942()
            .ok_or_else(|| {
                "deepseek4: loaded gfx942 backend cannot execute on this GPU".to_owned()
            })?
            .grouped_olora_e8(a, x, y, groups, m, k)
            .map_err(|e| format!("gfx942 grouped O-LoRA E8: {e:?}"))
    }

    pub(super) fn indexer_top_k_buf_parallel(
        self,
        gpu: &mut Gpu,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        n_idx_heads: i32,
        max_k: i32,
        bounded: bool,
    ) -> Result<(), String> {
        gpu.try_gfx942()
            .ok_or_else(|| {
                "deepseek4: loaded gfx942 backend cannot execute on this GPU".to_owned()
            })?
            .indexer_top_k_buf_parallel(
                scores,
                top_indices,
                n_compressed_buf,
                k_buf,
                n_idx_heads,
                max_k,
                bounded,
            )
            .map_err(|e| format!("gfx942 indexer top-k: {e:?}"))
    }
}

impl MoeBiasAwareMq2Backend for Gfx942Backend {
    fn gate_up(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        _nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String> {
        gpu.try_gfx942()
            .ok_or_else(|| {
                "deepseek4: loaded gfx942 backend cannot execute MQ2 gate-up on this GPU".to_owned()
            })?
            .mq2_lloyd_moe_gate_up_wave64(
                expert_ptrs,
                topk_indices,
                x_rot,
                y_gate,
                y_up,
                m,
                k,
                k_top,
            )
            .map_err(|e| format!("gfx942 MQ2 gate-up: {e:?}"))
    }

    fn rotate_x_batched(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        gpu.try_gfx942()
            .ok_or_else(|| {
                "deepseek4: loaded gfx942 backend cannot execute MQ rotation on this GPU".to_owned()
            })?
            .mq_rotate_x_wave64_batched(x, x_rot, k, batch_size)
            .map_err(|e| format!("gfx942 MQ rotate: {e:?}"))
    }

    fn down_expanded(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        _ownership_ptrs: &GpuTensor,
        _nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        gpu.try_gfx942()
            .ok_or_else(|| {
                "deepseek4: loaded gfx942 backend cannot execute MQ2 down on this GPU".to_owned()
            })?
            .mq2_lloyd_moe_down_expanded_wave64(
                expert_ptrs,
                topk_indices,
                rot_batch,
                expert_outputs,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| format!("gfx942 MQ2 deterministic down: {e:?}"))
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
        gpu.try_gfx942()
            .ok_or_else(|| {
                "deepseek4: loaded gfx942 backend cannot execute MQ2 hash down on this GPU"
                    .to_owned()
            })?
            .mq2_lloyd_moe_down_residual_wave64(
                expert_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                residual,
                m,
                k,
                k_top,
            )
            .map_err(|e| format!("gfx942 MQ2 hash down: {e:?}"))
    }
}
