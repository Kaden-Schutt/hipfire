// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use rdna_compute::{Gpu, GpuTensor};

/// Model-owned proof that a frozen MQ2R dense tower belongs to exact gfx1100.
///
/// The proof is deliberately separate from `Gpu::arch_caps`: DS4 chooses this
/// backend only after validating the P3 tensor recipe, and every operation
/// reacquires `Gfx1100Device` before it can reach an exact-target kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Gfx1100Backend {
    _sealed: (),
}

impl Gfx1100Backend {
    pub(super) fn try_new(gpu: &mut Gpu) -> Option<Self> {
        gpu.try_gfx1100().map(|_| Self { _sealed: () })
    }

    pub(super) fn dense_e8(
        self,
        gpu: &mut Gpu,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        gpu.try_gfx1100()
            .ok_or_else(|| {
                "deepseek4: loaded gfx1100 backend cannot execute on this GPU".to_owned()
            })?
            .ds4_dense_e8(weight, x, y, m, k)
            .map_err(|error| format!("gfx1100 dense E8: {error:?}"))
    }

    pub(super) fn grouped_olora_e8(
        self,
        gpu: &mut Gpu,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        gpu.try_gfx1100()
            .ok_or_else(|| {
                "deepseek4: loaded gfx1100 backend cannot execute on this GPU".to_owned()
            })?
            .ds4_grouped_olora_e8(weight, x, y, groups, m, k)
            .map_err(|error| format!("gfx1100 grouped O-LoRA E8: {error:?}"))
    }
}
