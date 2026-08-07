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
//! Other super-ops ordinarily run **replicated** and unchanged. An architecture
//! may explicitly opt `Attend` into dense tensor parallelism through the
//! fail-closed `ForwardBindings` attention-TP hooks; the default remains false,
//! so Qwen, MiniMax, and every existing EP route retain replicated attention.
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
use hipfire_dispatch::pipeline::superop::{
    dispatch_super_op, ForwardBindings, LayerProgram, SuperOpKind,
};
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

fn all_reduce_sum_f32_decode(
    gpus: &mut Gpus,
    refs: &[&DeviceBuffer],
    count: usize,
) -> Result<(), DispatchError> {
    // Decode stays on RCCL: its tiny per-token reduce is already fast. The
    // peer-direct diagnostic remains available for both MoE and attention TP.
    static PEER_DECODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let use_peer = *PEER_DECODE.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_EP_PEER_ALLREDUCE_DECODE").as_deref() == Ok("1")
    });
    if use_peer {
        gpus.all_reduce_sum_f32_peer(refs, count).map_err(hip_err)
    } else {
        gpus.all_reduce_sum_f32(refs, count).map_err(hip_err)
    }
}

fn tp_peer_hc4_admitted<B: ForwardBindings>(gpus: &Gpus, bindings: &[B]) -> bool {
    gpus.devices.len() == 4
        && gpus.peer_access_enabled
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
        && bindings.iter().all(ForwardBindings::supports_tp_peer_hc4)
}

fn tp_peer_hc3_admitted<B: ForwardBindings>(gpus: &Gpus, bindings: &[B]) -> bool {
    gpus.devices.len() == 3
        && gpus.peer_access_enabled
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
        && bindings.iter().all(ForwardBindings::supports_tp_peer_hc3)
}

fn tp_peer_hc3_graph_sync_enabled(gpus: &Gpus) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_TP3_HC_GRAPH_SYNC").as_deref() == Ok("1")
    }) && gpus.devices.len() == 3
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201() && device.graphs.capture_mode)
}

fn tp3_peer_signals(signals: &[DeviceBuffer; 3], rank: usize) -> [&DeviceBuffer; 2] {
    match rank {
        0 => [&signals[1], &signals[2]],
        1 => [&signals[0], &signals[2]],
        2 => [&signals[0], &signals[1]],
        _ => unreachable!("TP3 rank must be 0..3"),
    }
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
    assert_eq!(
        bindings.len(),
        n,
        "run_layer_program_ep: bindings.len() != n_ranks"
    );
    assert_eq!(
        partials.len(),
        n,
        "run_layer_program_ep: partials.len() != n_ranks"
    );

    for op in program {
        if matches!(op.kind, SuperOpKind::Attend)
            && bindings.iter().any(ForwardBindings::attention_tp_enabled)
        {
            if !bindings.iter().all(ForwardBindings::attention_tp_enabled) {
                return Err(DispatchError::Hip(
                    "run_layer_program_ep: mixed attention-TP admission across ranks".into(),
                ));
            }

            // Each rank computes its local head/O-LoRA shard and stops before
            // the residual mix, leaving one hidden-width partial in the
            // architecture-owned attention output tensor.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                bindings[r].run_attend_ep(&mut gpus.devices[r], &ctx, &op.binding)?;
            }

            if tp_peer_hc3_admitted(gpus, bindings) {
                let peer_partials = bindings
                    .iter()
                    .map(|binding| {
                        let partial = binding.ep_attention_partial().ok_or_else(|| {
                            DispatchError::Hip(
                                "run_layer_program_ep: attention TP partial missing".into(),
                            )
                        })?;
                        Ok(GpuTensor {
                            buf: unsafe { partial.buf.alias() },
                            shape: partial.shape.clone(),
                            dtype: partial.dtype,
                        })
                    })
                    .collect::<Result<Vec<_>, DispatchError>>()?;
                let peers = [&peer_partials[0], &peer_partials[1], &peer_partials[2]];
                if tp_peer_hc3_graph_sync_enabled(gpus) {
                    let (epoch, signals) = gpus.take_tp3_graph_fused_barrier().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_attend_peer_hc3_graph_sync(
                            &mut gpus.devices[r],
                            peers,
                            &signals[r],
                            tp3_peer_signals(&signals, r),
                            epoch,
                        )?;
                    }
                } else {
                    gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_attend_peer_hc3(&mut gpus.devices[r], peers)?;
                    }
                }
            } else if tp_peer_hc4_admitted(gpus, bindings) {
                // Borrow-independent aliases let the architecture hooks
                // consume all four peer pointers while each binding is
                // mutably advanced through its own HC residual mix.
                let peer_partials = bindings
                    .iter()
                    .map(|binding| {
                        let partial = binding.ep_attention_partial().ok_or_else(|| {
                            DispatchError::Hip(
                                "run_layer_program_ep: attention TP partial missing".into(),
                            )
                        })?;
                        Ok(GpuTensor {
                            buf: unsafe { partial.buf.alias() },
                            shape: partial.shape.clone(),
                            dtype: partial.dtype,
                        })
                    })
                    .collect::<Result<Vec<_>, DispatchError>>()?;
                let peers = [
                    &peer_partials[0],
                    &peer_partials[1],
                    &peer_partials[2],
                    &peer_partials[3],
                ];
                gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend_peer_hc4(&mut gpus.devices[r], peers)?;
                }
            } else {
                // Sum the input-column-sharded output projection directly in
                // its destination tensor. No staging copy or extra scratch.
                let refs: Vec<&DeviceBuffer> = bindings
                    .iter()
                    .map(|binding| {
                        binding
                            .ep_attention_partial()
                            .map(|partial| &partial.buf)
                            .ok_or_else(|| {
                                DispatchError::Hip(
                                    "run_layer_program_ep: attention TP partial missing".into(),
                                )
                            })
                    })
                    .collect::<Result<_, _>>()?;
                all_reduce_sum_f32_decode(gpus, &refs, residual_dim)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend(&mut gpus.devices[r])?;
                }
            }
        } else if matches!(op.kind, SuperOpKind::Moe) {
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

            if tp_peer_hc3_admitted(gpus, bindings) {
                let peers = [&partials[0], &partials[1], &partials[2]];
                if tp_peer_hc3_graph_sync_enabled(gpus) {
                    let (epoch, signals) = gpus.take_tp3_graph_fused_barrier().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_moe_peer_hc3_graph_sync(
                            &mut gpus.devices[r],
                            peers,
                            &signals[r],
                            tp3_peer_signals(&signals, r),
                            epoch,
                        )?;
                    }
                } else {
                    gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_moe_peer_hc3(&mut gpus.devices[r], peers)?;
                    }
                }
            } else if tp_peer_hc4_admitted(gpus, bindings) {
                let peers = [&partials[0], &partials[1], &partials[2], &partials[3]];
                gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_moe_peer_hc4(&mut gpus.devices[r], peers)?;
                }
            } else {
                // 3. All-reduce-sum the partials across ranks (in-place, RCCL).
                let refs: Vec<&DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
                all_reduce_sum_f32_decode(gpus, &refs, residual_dim)?;

                // 4. Fold the reduced partial into each residual stream.
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
                }
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
