// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Expert-parallel (EP) executor for the Ship 6 super-op substrate.
//!
//! Runs a lowered [`LayerProgram`] **replicated across N ranks** (every rank
//! runs every op on full, replicated attention/dense weights), special-casing
//! the `Moe` super-op with **all-reduce EP**:
//!
//! 1. zero each rank's routed partial,
//! 2. each rank computes ONLY its owned experts (+ the shared expert on rank 0)
//!    into its partial via [`ForwardBindings::run_moe_ep`] (non-owned experts
//!    read load-time zero-dummy weights → contribute 0),
//! 3. `all_reduce_sum_f32` the partials across ranks (RCCL),
//! 4. each rank adds the reduced partial into its residual stream via
//!    [`ForwardBindings::ep_add_into_residual`].
//!
//! All other super-ops (Attend / Norm / Proj / ResidualGemv / Recurrent / Conv
//! / Escape) run **replicated** and unchanged — every rank holds the full
//! weights and full KV, so they are deterministic functions of replicated
//! inputs and stay bit-identical across ranks. This is why EP needs no
//! attention-sharding (FaPhase) seam: the only divergence is at `Moe`.
//!
//! Ordering: every op (zero, run_moe_ep, the collective, the residual add, and
//! the next layer's ops) is enqueued on each device's `active_stream`, which is
//! FIFO — so the per-rank sequence is correctly ordered without host syncs
//! between ops or layers. The decode driver syncs once at the end before
//! reading logits.
//!
//! This executor drives ONE layer's program across all ranks; the per-arch EP
//! driver loops layers (advancing each rank's per-layer binding state) the same
//! way the single-GPU lowered driver loops `run_layer_program`.

use crate::pipeline::superop::{ForwardBindings, LayerProgram};
use crate::types::DispatchError;
use hipfire_hardware::{DeviceMesh, Gpus};
use rdna_compute::GpuTensor;

pub use crate::pipeline::superop::ensure_rank_streams;

/// Execute one lowered layer program across `gpus.devices.len()` EP ranks.
///
/// - `bindings[r]` drives rank `r`'s forward (it holds that rank's state /
///   weights / per-layer counters by reference, exactly like the single-GPU
///   `ForwardBindings` impl).
/// - `partials[r]` is rank `r`'s zeroed routed-output scratch, a contiguous f32
///   buffer of length `residual_dim` on `gpus.devices[r]`. The executor owns the
///   zero/all-reduce/add lifecycle; the binding only writes its owned-expert
///   contribution into it during `run_moe_ep`.
/// - `residual_dim` is the residual width (= hidden size) used for the partial
///   memset byte size and the all-reduce element count.
///
/// Every device must have an `active_stream` set ([`ensure_rank_streams`]).
pub fn run_layer_program_ep<B: ForwardBindings>(
    mesh: &DeviceMesh,
    gpus: &mut Gpus,
    bindings: &mut [B],
    partials: &[GpuTensor],
    program: &LayerProgram,
    residual_dim: usize,
) -> Result<(), DispatchError> {
    debug_assert_eq!(
        partials[0].numel(),
        residual_dim,
        "ep shim: residual_dim mismatch"
    );
    crate::pipeline::superop::run_layer_program_mesh(mesh, gpus, bindings, partials, program)
}
