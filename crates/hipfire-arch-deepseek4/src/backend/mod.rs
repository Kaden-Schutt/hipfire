// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek4 execution backends.
//!
//! Backend identity belongs to the loaded model weights, not the process-wide
//! GPU context. This prevents a model swap or a Qwen load from inheriting DS4
//! architecture policy.

mod gfx1201;
mod gfx942;

use hipfire_dispatch::families::moe::MoeBiasAwareMq2Backend;
use rdna_compute::{Gpu, GpuTensor};

use gfx1201::Gfx1201Backend;
use gfx942::Gfx942Backend;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Mq2rBackend {
    #[default]
    Portable,
    Gfx1151,
    Gfx1201(Gfx1201Backend),
    Gfx942(Gfx942Backend),
}

impl Mq2rBackend {
    /// Select a backend only after the loader has verified the frozen MQ2R P3
    /// tensor recipe. Exact device checks, rather than a broad architecture
    /// family or environment variable, grant native eligibility.
    pub(crate) fn for_verified_mq2r(gpu: &mut Gpu) -> Self {
        if gpu.arch_caps.is_gfx1151() {
            Self::Gfx1151
        } else if let Some(gfx1201) = Gfx1201Backend::try_new(gpu) {
            Self::Gfx1201(gfx1201)
        } else if let Some(gfx942) = Gfx942Backend::try_new(gpu) {
            Self::Gfx942(gfx942)
        } else {
            Self::Portable
        }
    }

    pub(crate) const fn is_gfx942(self) -> bool {
        matches!(self, Self::Gfx942(_))
    }

    pub(crate) const fn is_gfx1151(self) -> bool {
        matches!(self, Self::Gfx1151)
    }

    pub(crate) const fn is_gfx1201(self) -> bool {
        matches!(self, Self::Gfx1201(_))
    }

    /// The certified gfx1151 route uses the fused atomic MoE down projection.
    /// Other backends stay on expanded output plus a fixed-order combine until
    /// their own arithmetic route is certified.
    pub(crate) const fn uses_atomic_moe_down(self) -> bool {
        self.is_gfx1151()
    }

    /// Native MQ2 decode operations are a capability of the model-owned exact
    /// gfx942 backend. Portable and gfx1151 models never expose this object to
    /// the shared MoE executor.
    pub(crate) fn bias_aware_native_backend(&self) -> Option<&dyn MoeBiasAwareMq2Backend> {
        match self {
            Self::Gfx1201(backend) => Some(backend),
            Self::Gfx942(backend) => Some(backend),
            Self::Portable | Self::Gfx1151 => None,
        }
    }

    pub(crate) fn grouped_olora_e8(
        self,
        gpu: &mut Gpu,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        match self {
            Self::Gfx942(backend) => backend.grouped_olora_e8(gpu, a, x, y, groups, m, k),
            Self::Portable | Self::Gfx1151 | Self::Gfx1201(_) => {
                Err("deepseek4: gfx942 grouped O-LoRA requested by a non-gfx942 backend".to_owned())
            }
        }
    }

    /// Run the exact-gfx942 indexer when this verified model owns that
    /// backend. `bounded` forwards the caller's kernel selection (false: O(N^2)
    /// reference, true: opt-in O(N log^2 K) bounded bitonic); no defaulting
    /// happens here. `Ok(false)` leaves portable/gfx1151 selection to the
    /// caller.
    pub(crate) fn try_indexer_top_k_buf_parallel(
        self,
        gpu: &mut Gpu,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        n_idx_heads: i32,
        max_k: i32,
        bounded: bool,
    ) -> Result<bool, String> {
        match self {
            Self::Gfx942(backend) => {
                backend.indexer_top_k_buf_parallel(
                    gpu,
                    scores,
                    top_indices,
                    n_compressed_buf,
                    k_buf,
                    n_idx_heads,
                    max_k,
                    bounded,
                )?;
                Ok(true)
            }
            Self::Portable | Self::Gfx1151 | Self::Gfx1201(_) => Ok(false),
        }
    }
}
