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

use crate::multi_gpu::Gpus;
use hip_bridge::{DeviceBuffer, HipError};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{dispatch_super_op, ForwardBindings, LayerProgram, SuperOpKind};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::GpuTensor;

fn hip_err(e: HipError) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

/// Ensure every device owns an `active_stream` (the stream the EP collectives
/// and per-rank work run on). Idempotent; safe to call before each layer.
pub fn ensure_rank_streams(gpus: &mut Gpus) -> Result<(), DispatchError> {
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().map_err(hip_err)?;
        if dev.active_stream.is_none() {
            dev.active_stream = Some(dev.hip.stream_create().map_err(hip_err)?);
        }
    }
    Ok(())
}

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
    gpus: &mut Gpus,
    bindings: &mut [B],
    partials: &[GpuTensor],
    program: &LayerProgram,
    residual_dim: usize,
) -> Result<(), DispatchError> {
    let n = gpus.devices.len();
    assert_eq!(bindings.len(), n, "run_layer_program_ep: bindings.len() != n_ranks");
    assert_eq!(partials.len(), n, "run_layer_program_ep: partials.len() != n_ranks");

    for op in program {
        if matches!(op.kind, SuperOpKind::Moe) {
            // 1. Zero each rank's routed partial on its own stream.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let stream = gpus.devices[r]
                    .active_stream
                    .as_ref()
                    .ok_or_else(|| DispatchError::Hip(format!(
                        "run_layer_program_ep: device {r} has no active_stream (call ensure_rank_streams)"
                    )))?;
                gpus.devices[r]
                    .hip
                    .memset_async(&partials[r].buf, 0, residual_dim * 4, stream)
                    .map_err(hip_err)?;
            }

            // 2. Each rank computes its owned-expert routed partial (+ shared on
            //    rank 0 via skip_shared=false; ranks>0 skip the shared down).
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                bindings[r].run_moe_ep(
                    &mut gpus.devices[r],
                    &ctx,
                    &op.binding,
                    &partials[r],
                    /* skip_shared = */ r != 0,
                )?;
            }

            // 3. All-reduce-sum the partials across ranks (in-place, RCCL).
            //    Decode stays on RCCL: its tiny per-token reduce is already fast
            //    (NOT the bottleneck — measured 51.4 RCCL vs 48.0 peer-direct on
            //    MiniMax 62-layer decode), peer-direct's per-layer wait_boundary
            //    host-sync only adds overhead, and RCCL preserves qwen35's
            //    validated byte-identical decode. Peer-direct is the win for
            //    PREFILL (large batched reduce), where RCCL is ~40 ms/call —
            //    that path uses all_reduce_sum_f32_peer directly. Opt decode into
            //    peer-direct with HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 if needed.
            let refs: Vec<&DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
            static PEER_DECODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let use_peer = *PEER_DECODE
                .get_or_init(|| std::env::var("HIPFIRE_EP_PEER_ALLREDUCE_DECODE").as_deref() == Ok("1"));
            if use_peer {
                gpus.all_reduce_sum_f32_peer(&refs, residual_dim).map_err(hip_err)?;
            } else {
                gpus.all_reduce_sum_f32(&refs, residual_dim).map_err(hip_err)?;
            }

            // 4. Each rank adds the reduced partial into its residual stream.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
            }
        } else {
            // Replicated op — every rank runs it unchanged on full weights.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                dispatch_super_op(&mut gpus.devices[r], &ctx, op, &mut bindings[r])?;
            }
        }
    }
    Ok(())
}
