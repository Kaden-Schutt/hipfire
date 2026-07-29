// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fail-closed integration gate for Redline record/replay.
//!
//! This module records the central HIP launch surface during warmup and owns
//! the fail-closed selection state. It deliberately does not reinterpret
//! `void**` arguments: a model adapter must supply explicit resource accesses
//! and a kernarg ABI to `redline-dispatch` before installing a prepared plan.
//! Replay remains default-off except for the runtime automatic default
//! `a3b_mq4r_redline_default` (exact `gfx1100`/`gfx1151`/`gfx1201`, single-GPU
//! Qwen A3B `arch_id == 6`, case-insensitive `.mq4r`; `gfx1200`/others opt-in)
//! selected by the daemon after model load. Runtime default ≠ Redline
//! certification/registry admission; built-in `hip` profile or explicit backend
//! selection disables the automatic default.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use hip_bridge::HipRuntime;
use redline_dispatch::aql::{
    load_symbols, BatchFencePolicy, Executable, Gfx10DispatchInitiatorPolicy,
    Gfx10Pm4CommandBuffer, Gfx11ComputeResourceLimitsPolicy, Gfx11DispatchInterleave,
    Gfx12Pm4CommandBuffer, GpuBatchTiming, GpuDevice, GpuMultiQueueTiming, GpuSelector,
    HeaderPolicy, KernargBuffer, KernargPool, Kernel, LaunchGeometry, PhasedMultiQueuePm4Ib,
    QueuePolicy, RecordedDispatch, Runtime, SingleQueueBatchGraph, SingleQueuePm4Ib,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayBackendRequest {
    Hip,
    Shadow,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayTransport {
    AqlPackets,
    Pm4Ib,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4Architecture {
    Gfx10,
    Gfx11,
    Gfx12,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gfx11EntryAcquirePolicy {
    System,
    Agent,
    Vmem,
    None,
}

impl Pm4Architecture {
    fn from_device(device: &GpuDevice) -> Result<Self, String> {
        let name = device.name().to_ascii_lowercase();
        if name.starts_with("gfx10") {
            Ok(Self::Gfx10)
        } else if name.starts_with("gfx11") {
            Ok(Self::Gfx11)
        } else if name.starts_with("gfx12") {
            Ok(Self::Gfx12)
        } else {
            Err(format!(
                "retained PM4 has no certified register map for HSA agent {:?}",
                device.name()
            ))
        }
    }
}

#[derive(Clone)]
enum Pm4Commands {
    Legacy {
        architecture: Pm4Architecture,
        commands: Gfx10Pm4CommandBuffer,
    },
    Gfx12(Gfx12Pm4CommandBuffer),
}

fn create_phased_pm4_graph(
    architecture: Pm4Architecture,
    device: &GpuDevice,
    pool: &KernargPool,
    phases: &[Vec<Pm4Commands>],
    native_sync: bool,
) -> Result<PhasedMultiQueuePm4Ib, String> {
    match architecture {
        Pm4Architecture::Gfx10 | Pm4Architecture::Gfx11 => {
            let legacy = phases
                .iter()
                .map(|phase| {
                    phase
                        .iter()
                        .map(|commands| match commands {
                            Pm4Commands::Legacy {
                                architecture: actual,
                                commands,
                            } if *actual == architecture => Ok(commands.clone()),
                            _ => Err("mixed PM4 architecture in phased graph".to_owned()),
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            match architecture {
                Pm4Architecture::Gfx10 if native_sync => {
                    PhasedMultiQueuePm4Ib::create_profiled_native_gfx10(device, pool, &legacy)
                }
                Pm4Architecture::Gfx10 => {
                    PhasedMultiQueuePm4Ib::create_profiled_gfx10(device, pool, &legacy)
                }
                Pm4Architecture::Gfx11 if native_sync => {
                    PhasedMultiQueuePm4Ib::create_profiled_native_gfx11(device, pool, &legacy)
                }
                Pm4Architecture::Gfx11 => {
                    PhasedMultiQueuePm4Ib::create_profiled_gfx11(device, pool, &legacy)
                }
                Pm4Architecture::Gfx12 => unreachable!(),
            }
        }
        Pm4Architecture::Gfx12 => {
            if native_sync {
                return Err(
                    "native PM4 phase synchronization is not yet lowered for gfx12".to_owned(),
                );
            }
            let gfx12 = phases
                .iter()
                .map(|phase| {
                    phase
                        .iter()
                        .map(|commands| match commands {
                            Pm4Commands::Gfx12(commands) => Ok(commands.clone()),
                            Pm4Commands::Legacy { .. } => {
                                Err("mixed PM4 architecture in phased graph".to_owned())
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            PhasedMultiQueuePm4Ib::create_profiled(device, pool, &gfx12)
        }
    }
    .map_err(|error| error.to_string())
}

impl Pm4Commands {
    fn new(
        architecture: Pm4Architecture,
        policy: Pm4RegisterPolicy,
        dispatch_initiator_policy: Gfx10DispatchInitiatorPolicy,
        dispatch_interleave: Option<Gfx11DispatchInterleave>,
        resource_limits_policy: Gfx11ComputeResourceLimitsPolicy,
    ) -> Self {
        match architecture {
            Pm4Architecture::Gfx10 | Pm4Architecture::Gfx11 => {
                let commands = match policy {
                    Pm4RegisterPolicy::Legacy => Gfx10Pm4CommandBuffer::new(),
                    Pm4RegisterPolicy::Static | Pm4RegisterPolicy::Stateful => {
                        Gfx10Pm4CommandBuffer::new_stateful()
                    }
                }
                .with_dispatch_initiator_policy(dispatch_initiator_policy)
                .with_dispatch_interleave(dispatch_interleave)
                .with_resource_limits_policy(resource_limits_policy);
                Self::Legacy {
                    architecture,
                    commands,
                }
            }
            Pm4Architecture::Gfx12 => {
                let commands = match policy {
                    Pm4RegisterPolicy::Legacy => Gfx12Pm4CommandBuffer::new(),
                    Pm4RegisterPolicy::Static => Gfx12Pm4CommandBuffer::new_static_stateful(),
                    Pm4RegisterPolicy::Stateful => Gfx12Pm4CommandBuffer::new_stateful(),
                };
                Self::Gfx12(commands)
            }
        }
    }

    fn acquire_entry(&mut self, gfx12_gcr_trim: bool, gfx11_policy: Gfx11EntryAcquirePolicy) {
        match self {
            Self::Legacy { commands, .. } => match gfx11_policy {
                Gfx11EntryAcquirePolicy::System => commands.acquire_system(),
                Gfx11EntryAcquirePolicy::Agent => commands.acquire_inter_node_same_agent(),
                Gfx11EntryAcquirePolicy::Vmem => commands.acquire_inter_node_vmem(),
                Gfx11EntryAcquirePolicy::None => {}
            },
            Self::Gfx12(commands) if gfx12_gcr_trim => commands.acquire_system_gfx12(),
            Self::Gfx12(commands) => commands.acquire_system(),
        }
    }

    fn acquire_inter_node(&mut self, gfx12_gcr_trim: bool, vmem_only: bool) {
        match self {
            Self::Legacy { commands, .. } if vmem_only => commands.acquire_inter_node_vmem(),
            Self::Legacy { commands, .. } => commands.acquire_inter_node_same_agent(),
            Self::Gfx12(commands) if gfx12_gcr_trim => commands.acquire_inter_node_gfx12(),
            Self::Gfx12(commands) => commands.acquire_system(),
        }
    }

    fn requires_dependency_acquire(&self) -> bool {
        matches!(self, Self::Legacy { .. })
    }

    fn wait_compute_idle(&mut self) {
        match self {
            Self::Legacy { commands, .. } => commands.wait_compute_idle(),
            Self::Gfx12(commands) => commands.wait_compute_idle(),
        }
    }

    fn dispatch(
        &mut self,
        kernel: &Kernel,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut std::ffi::c_void,
    ) -> Result<(), String> {
        match self {
            Self::Legacy { commands, .. } => commands
                .dispatch(kernel, geometry, dynamic_group_bytes, kernarg_address)
                .map_err(|error| error.to_string()),
            Self::Gfx12(commands) => commands
                .dispatch(kernel, geometry, dynamic_group_bytes, kernarg_address)
                .map_err(|error| error.to_string()),
        }
    }

    fn len_dwords(&self) -> u32 {
        match self {
            Self::Legacy { commands, .. } => commands.len_dwords(),
            Self::Gfx12(commands) => commands.len_dwords(),
        }
    }

    fn populate_dispatch_span_boundaries(
        &self,
        boundaries: &mut [Pm4DispatchBoundary],
    ) -> Result<(), String> {
        let Self::Gfx12(commands) = self else {
            return Err("per-dispatch PM4 profiling currently requires gfx12".to_owned());
        };
        let attributions = commands
            .dispatch_span_attributions()
            .map_err(|error| error.to_string())?;
        if attributions.len() != boundaries.len() {
            return Err(format!(
                "generated PM4 dispatch attribution mismatch: expected {}, got {}",
                boundaries.len(),
                attributions.len()
            ));
        }
        for (boundary, attribution) in boundaries.iter_mut().zip(attributions) {
            boundary.entry_acquire = attribution.entry_acquire;
            boundary.wait_compute_idle = attribution.wait_compute_idle;
            boundary.acquire_inter_node = attribution.acquire_inter_node;
        }
        Ok(())
    }

    fn create_graph(
        &self,
        device: &GpuDevice,
        pool: &KernargPool,
        ib_pool: Option<&KernargPool>,
        cu_mask: Option<&[u32; 2]>,
        dispatch_profile: bool,
    ) -> Result<SingleQueuePm4Ib, String> {
        let graph = match self {
            Self::Legacy {
                architecture: Pm4Architecture::Gfx10,
                commands,
            } => SingleQueuePm4Ib::create_profiled_gfx10(device, pool, commands),
            Self::Legacy {
                architecture: Pm4Architecture::Gfx11,
                commands,
            } => SingleQueuePm4Ib::create_profiled_gfx11(device, pool, commands),
            Self::Legacy {
                architecture: Pm4Architecture::Gfx12,
                ..
            } => unreachable!("gfx12 never uses the legacy PM4 command variant"),
            Self::Gfx12(commands) => {
                // HIPFIRE_REDLINE_DISPATCH_PROFILE=1 builds a tape carrying one
                // GPU-clock write per dispatch, so a machine whose retained path
                // underperforms can report WHERE the time goes instead of only
                // how much there is. The instrumented tape necessarily has a
                // different dword count and sequence hash, so such a run cannot
                // satisfy a golden fixture — it is a diagnostic, not a
                // certification.
                if dispatch_profile || dispatch_profile_enabled() {
                    SingleQueuePm4Ib::create_dispatch_profiled(device, pool, commands)
                } else if let Some(ib_pool) = ib_pool {
                    SingleQueuePm4Ib::create_profiled_with_ib_pool(device, pool, ib_pool, commands)
                } else {
                    SingleQueuePm4Ib::create_profiled(device, pool, commands)
                }
            }
        }
        .map_err(|error| error.to_string())?;
        if let Some(cu_mask) = cu_mask {
            graph
                .set_cu_mask(64, cu_mask)
                .map_err(|error| error.to_string())?;
        }
        Ok(graph)
    }
}

/// Candidate-only cache classification produced by Radiowave inspection of
/// the exact gfx11 code objects captured by the MQ4R replay tapes. This list is
/// deliberately gated by `HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE`: the legacy
/// Hipfire JIT path does not yet emit hash-bound Radiowave manifests, so the
/// safe default remains scalar-or-unknown until that binding is added.
fn radiowave_vmem_only_consumer(kernel: &str) -> bool {
    matches!(
        kernel,
        "conv1d_silu_split_f32"
            | "conv1d_silu_split_qknorm_b256"
            | "deinterleave_f32"
            | "fused_qk_l2_norm_scale_f32"
            | "fused_qkv_hfq4g256_k2048_all_buffer_gfx1151"
            | "fused_qkv_hfq4g256_k2048_all_buffer_slc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151"
            | "fused_rmsnorm_mq_rotate"
            | "fused_rmsnorm_mq_rotate_vecsum"
            | "fused_sigmoid_alpha_gate_f32"
            | "fused_silu_mul_mq_rotate"
            | "gated_norm_f32"
            | "gated_norm_mq_rotate_gfx1100"
            | "gated_norm_mq_rotate_gfx1151"
            | "gemv_hfq4g256_residual_rt_low_gfx1151"
            | "moe_router_softmax_topk_k8_wave64_exact"
            | "moe_topk_renorm_k8"
            | "mq_rotate_x"
            | "repeat_interleave_qk_f32"
            | "rmsnorm_f32"
            | "sigmoid_mul_f32"
    )
}

fn pm4_vmem_acquire_enabled(architecture: Pm4Architecture, configured: bool, kernel: &str) -> bool {
    architecture != Pm4Architecture::Gfx12 && configured && radiowave_vmem_only_consumer(kernel)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecordedAccessMode {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RecordedResourceAccess {
    allocation_base: u64,
    allocation_bytes: u64,
    // Diagnostic pointer start within the allocation. Scheduling remains
    // allocation-wide; this proves whether a blocked boundary has any exact
    // producer/consumer pointer dependency before byte ranges are considered.
    access_base: u64,
    mode: RecordedAccessMode,
}

impl RecordedResourceAccess {
    fn end(self) -> u64 {
        self.allocation_base + self.allocation_bytes
    }

    fn conflicts(self, other: Self) -> bool {
        let overlaps = self.allocation_base < other.end() && other.allocation_base < self.end();
        overlaps
            && (self.mode == RecordedAccessMode::Write || other.mode == RecordedAccessMode::Write)
    }

    fn same_start_conflicts(self, other: Self) -> bool {
        self.access_base == other.access_base
            && (self.mode == RecordedAccessMode::Write || other.mode == RecordedAccessMode::Write)
    }
}

#[derive(Clone, Copy)]
struct PointerEffect {
    offset: usize,
    mode: RecordedAccessMode,
}

const fn read(offset: usize) -> PointerEffect {
    PointerEffect {
        offset,
        mode: RecordedAccessMode::Read,
    }
}

const fn write(offset: usize) -> PointerEffect {
    PointerEffect {
        offset,
        mode: RecordedAccessMode::Write,
    }
}

/// Pointer fields and memory effects for kernels admitted to Qwen AR replay.
///
/// A non-const kernel pointer is conservatively classified as `Write`, which
/// also covers read-modify-write effects. Unknown kernels fail closed and keep
/// their compute-idle boundaries. Offsets are the naturally aligned HIP
/// kernarg ABI offsets verified by the captured-blob/loader parity gate.
fn pointer_effects(kernel: &str) -> Option<Vec<PointerEffect>> {
    if matches!(
        kernel,
        "fused_gate_up_hfq4g256" | "fused_gate_up_hfq4g256_k1024_gfx1201"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151"
    ) {
        // The split producers share read-only routing, activation, and packed
        // expert allocations but write distinct gate/up output allocations.
        return Some(vec![read(0), read(8), read(16), write(24)]);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_low_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_route_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }
    if matches!(
        kernel,
        "fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151"
    ) {
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
        ]);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_hybrid_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_clustered_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24)]);
    }
    if kernel == "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate" {
        return Some(vec![
            read(0),
            write(8),
            write(16),
            read(32),
            read(40),
            read(48),
            read(56),
            write(64),
        ]);
    }
    if kernel == "moe_router_softmax_topk_k8_wave64"
        || kernel == "moe_router_softmax_topk_k8_wave64_exact"
    {
        return Some(vec![read(0), write(8), write(16)]);
    }
    if kernel.starts_with("gated_delta_net_q8_compact2_") {
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(80),
        ]);
    }
    if kernel == "fused_qkvza_hfq4g256_k2048_scalar_prep" {
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
            read(72),
            read(80),
        ]);
    }
    if kernel == "conv1d_silu_split_qknorm_b256_scalar_prep" {
        return Some(vec![
            write(0),
            write(8),
            write(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            read(64),
            read(72),
        ]);
    }
    if kernel.starts_with("conv1d_silu_split_qknorm_") {
        return Some(vec![
            write(0),
            write(8),
            write(16),
            read(24),
            read(32),
            write(40),
        ]);
    }
    match kernel {
        "fused_rmsnorm_mq_rotate"
        | "fused_rmsnorm_mq_rotate_vecsum"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_const"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_lds" => {
            Some(vec![read(0), read(8), read(16), read(24), write(32)])
        }
        "fused_rmsnorm_mq_rotate_wavegrid" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
        ]),
        "rmsnorm_reduce_gfx1100" => Some(vec![read(0), write(8)]),
        "rotate_with_rms_gfx1100" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]),
        "fused_qkvza_hfq4g256"
        | "fused_qkvza_hfq4g256_k2048"
        | "fused_qkvza_hfq4g256_k2048_r2"
        | "fused_qkvza_hfq4g256_k2048_cpol_slc"
        | "fused_qkvza_hfq4g256_wavepack4"
        | "fused_qkvza_hfq4g256_ldsx8"
        | "fused_qkvza_hfq4g256_reduce_chain" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
        ]),
        "fused_sigmoid_alpha_gate_f32" => Some(vec![write(0), write(8), read(16), read(24)]),
        "conv1d_silu_split_f32" => Some(vec![
            write(0),
            write(8),
            write(16),
            read(24),
            read(32),
            write(40),
        ]),
        "fused_qk_l2_norm_scale_f32" => Some(vec![write(0), write(8)]),
        "repeat_interleave_qk_f32" => Some(vec![read(0), read(8), write(16), write(24)]),
        "gated_delta_net_q8_fast" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(80),
        ]),
        "gated_norm_f32" => Some(vec![read(0), read(8), read(16), write(24)]),
        "gated_norm_mq_rotate_gfx1100" | "gated_norm_mq_rotate_gfx1151" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]),
        "qwen35_fa_prep_gfx1100" | "qwen35_fa_prep_gfx1151" => Some(vec![
            read(0),
            write(8),
            write(16),
            write(24),
            read(32),
            read(40),
            read(48),
        ]),
        "kv_cache_write_q8_0_pair" => Some(vec![write(0), write(8), read(16), read(24), read(32)]),
        "mq_rotate_x" => Some(vec![read(0), write(8), read(16), read(24)]),
        "gemv_hfq4g256"
        | "gemv_hfq4g256_lm_head_dot2_gfx1151"
        | "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"
        | "gemv_hfq4g256_k2048"
        | "gemv_hfq4g256_residual"
        | "gemv_hfq4g256_residual_cpol_rt"
        | "gemv_hfq4g256_residual_cpol_rt_low"
        | "gemv_hfq4g256_residual_cpol_slc"
        | "gemv_hfq4g256_residual_k2048"
        | "gemv_hfq4g256_residual_k4096_gfx1151"
        | "gemv_hfq4g256_residual_multirow_r2_gfx1151"
        | "gemv_hfq4g256_residual_rt_low_gfx1151"
        | "gemv_hfq4g256_residual_wave64"
        | "gemv_hfq4g256_wide"
        | "gemv_hfq4g256_multirow_r2"
        | "gemv_hfq4g256_multirow_r4"
        | "gemv_hfq4g256_multirow_r8" => Some(vec![read(0), read(8), write(16)]),
        "softmax_f32" => Some(vec![write(0)]),
        "moe_topk_renorm_k8" => Some(vec![read(0), write(8), write(16)]),
        "fused_silu_mul_mq_rotate" => Some(vec![read(0), read(8), read(16), read(24), write(32)]),
        "gemv_hfq4g256_residual_sigmoid_scaled_gpu" => {
            Some(vec![read(0), read(8), write(16), read(24)])
        }
        "gemv_hfq4g256_moe_gate_up_k8_indexed"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2" => {
            Some(vec![read(0), read(8), read(16), write(24), write(32)])
        }
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded"
        | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc" => {
            Some(vec![read(0), read(8), read(16), write(24)])
        }
        "gemv_hfq4g256_moe_down_k8_indexed_last_combine" => Some(vec![
            read(0),
            read(8),
            read(16),
            write(24),
            read(32),
            write(40),
        ]),
        "moe_down_combine_k8_batched" | "moe_down_combine_k8_batched_vec4" => {
            Some(vec![read(0), read(8), write(16)])
        }
        "moe_down_combine_rmsnorm_mq_rotate_vecsum"
        | "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151" => Some(vec![
            read(0),
            read(8),
            write(16),
            read(24),
            read(32),
            read(40),
            write(48),
        ]),
        "fused_qkv_hfq4g256" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
            write(48),
        ]),
        "deinterleave_f32" => Some(vec![read(0), write(8), write(16)]),
        "rmsnorm_f32" => Some(vec![read(0), read(8), write(16)]),
        "rope_partial_halfsplit_f32" => Some(vec![write(0), write(8), read(16)]),
        "kv_cache_write_asym_k_fwht3" => {
            Some(vec![write(0), read(8), read(16), read(24), read(32)])
        }
        "kv_cache_write_q8_0" => Some(vec![write(0), read(8), read(16)]),
        "attention_flash_fwht3_tile" => Some(vec![
            read(0),
            read(8),
            read(16),
            write(24),
            read(32),
            read(40),
            read(48),
        ]),
        "attention_flash_q8_0_tile" => Some(vec![read(0), read(8), read(16), write(24), read(32)]),
        "attention_flash_q8_0_reduce" => Some(vec![read(0), write(8), read(24)]),
        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151" => Some(vec![
            read(0),
            write(8),
            read(16),
            read(24),
            read(32),
            read(48),
        ]),
        "sigmoid_mul_f32" => Some(vec![write(0), read(8)]),
        _ => None,
    }
}

fn expected_kernarg_bytes(kernel: &str) -> Option<usize> {
    if matches!(
        kernel,
        "fused_gate_up_hfq4g256" | "fused_gate_up_hfq4g256_k1024_gfx1201"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151"
    ) {
        return Some(48);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_low_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_route_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"
    ) {
        return Some(48);
    }
    if matches!(
        kernel,
        "fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151"
    ) {
        return Some(96);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_hybrid_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_clustered_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151"
    ) {
        return Some(48);
    }
    if kernel.starts_with("gated_delta_net_q8_compact2_") {
        return Some(96);
    }
    if kernel == "conv1d_silu_split_qknorm_b256_scalar_prep" {
        return Some(112);
    }
    if kernel.starts_with("conv1d_silu_split_qknorm_") {
        return Some(80);
    }
    if kernel == "fused_qkvza_hfq4g256_k2048_scalar_prep" {
        return Some(112);
    }
    if kernel == "moe_router_softmax_topk_k8_wave64"
        || kernel == "moe_router_softmax_topk_k8_wave64_exact"
    {
        return Some(32);
    }
    match kernel {
        "softmax_f32" => Some(16),
        "fused_qk_l2_norm_scale_f32"
        | "gemv_hfq4g256"
        | "gemv_hfq4g256_lm_head_dot2_gfx1151"
        | "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"
        | "gemv_hfq4g256_k2048"
        | "gemv_hfq4g256_residual"
        | "gemv_hfq4g256_residual_cpol_rt"
        | "gemv_hfq4g256_residual_cpol_rt_low"
        | "gemv_hfq4g256_residual_cpol_slc"
        | "gemv_hfq4g256_residual_k2048"
        | "gemv_hfq4g256_residual_k4096_gfx1151"
        | "gemv_hfq4g256_residual_multirow_r2_gfx1151"
        | "gemv_hfq4g256_residual_rt_low_gfx1151"
        | "gemv_hfq4g256_residual_wave64"
        | "gemv_hfq4g256_wide"
        | "gemv_hfq4g256_multirow_r2"
        | "gemv_hfq4g256_multirow_r4"
        | "gemv_hfq4g256_multirow_r8"
        | "deinterleave_f32"
        | "kv_cache_write_q8_0"
        | "moe_down_combine_k8_batched"
        | "moe_down_combine_k8_batched_vec4"
        | "moe_topk_renorm_k8"
        | "rmsnorm_f32"
        | "rmsnorm_reduce_gfx1100"
        | "sigmoid_mul_f32" => Some(32),
        "attention_flash_q8_0_reduce"
        | "fused_rmsnorm_mq_rotate"
        | "fused_rmsnorm_mq_rotate_vecsum"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_const"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_lds"
        | "fused_sigmoid_alpha_gate_f32"
        | "fused_silu_mul_mq_rotate"
        | "gated_norm_f32"
        | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded"
        | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2"
        | "gemv_hfq4g256_residual_sigmoid_scaled_gpu"
        | "kv_cache_write_asym_k_fwht3"
        | "kv_cache_write_q8_0_pair"
        | "mq_rotate_x"
        | "repeat_interleave_qk_f32"
        | "rope_partial_halfsplit_f32" => Some(48),
        "conv1d_silu_split_f32"
        | "gated_norm_mq_rotate_gfx1100"
        | "gated_norm_mq_rotate_gfx1151"
        | "qwen35_fa_prep_gfx1100"
        | "qwen35_fa_prep_gfx1151"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151"
        | "fused_rmsnorm_mq_rotate_wavegrid"
        | "rotate_with_rms_gfx1100" => Some(64),
        "moe_down_combine_rmsnorm_mq_rotate_vecsum"
        | "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151" => Some(72),
        "gemv_hfq4g256_moe_down_k8_indexed_last_combine" => Some(64),
        "attention_flash_q8_0_tile"
        | "fused_qkv_hfq4g256"
        | "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate" => Some(80),
        "attention_flash_fwht3_tile"
        | "fused_qkvza_hfq4g256"
        | "fused_qkvza_hfq4g256_k2048"
        | "fused_qkvza_hfq4g256_k2048_r2"
        | "fused_qkvza_hfq4g256_k2048_cpol_slc"
        | "fused_qkvza_hfq4g256_wavepack4"
        | "fused_qkvza_hfq4g256_ldsx8"
        | "fused_qkvza_hfq4g256_reduce_chain"
        | "gated_delta_net_q8_fast" => Some(96),
        _ => None,
    }
}

fn recorded_resource_accesses(
    hip: &HipRuntime,
    kernel: &str,
    kernarg: &[u8],
) -> Option<Vec<RecordedResourceAccess>> {
    if std::mem::size_of::<usize>() != 8 {
        return None;
    }
    if kernarg.len() != expected_kernarg_bytes(kernel)? {
        return None;
    }
    let effects = pointer_effects(kernel)?;
    let mut accesses = BTreeMap::<(u64, u64), (u64, RecordedAccessMode)>::new();
    for effect in effects {
        let bytes: [u8; 8] = kernarg
            .get(effect.offset..effect.offset + 8)?
            .try_into()
            .ok()?;
        let address = u64::from_ne_bytes(bytes);
        if address == 0 {
            continue;
        }
        let (base, size) = hip.mem_get_address_range(address as usize as *mut _).ok()?;
        let base = base as usize as u64;
        let size = u64::try_from(size).ok()?;
        let entry = accesses
            .entry((base, address))
            .or_insert((size, effect.mode));
        if entry.0 != size {
            return None;
        }
        if effect.mode == RecordedAccessMode::Write {
            entry.1 = RecordedAccessMode::Write;
        }
    }
    Some(
        accesses
            .into_iter()
            .map(
                |((allocation_base, access_base), (allocation_bytes, mode))| {
                    RecordedResourceAccess {
                        allocation_base,
                        allocation_bytes,
                        access_base,
                        mode,
                    }
                },
            )
            .collect(),
    )
}

#[derive(Default)]
struct ResourceFrontier {
    accesses: Vec<RecordedResourceAccess>,
    known: bool,
}

impl ResourceFrontier {
    fn covered(&self, current: &RecordedHipLaunch) -> bool {
        self.known && current.accesses.is_some()
    }

    fn independent(&self, current: &RecordedHipLaunch) -> bool {
        let Some(current) = &current.accesses else {
            return false;
        };
        self.known
            && !self
                .accesses
                .iter()
                .any(|left| current.iter().any(|right| left.conflicts(*right)))
    }

    fn independent_by_exact_start(&self, current: &RecordedHipLaunch) -> bool {
        let Some(current) = &current.accesses else {
            return false;
        };
        self.known
            && !self.accesses.iter().any(|left| {
                current
                    .iter()
                    .any(|right| left.same_start_conflicts(*right))
            })
    }

    fn advance(&mut self, current: &RecordedHipLaunch, independent: bool) {
        if !independent {
            self.accesses.clear();
            self.known = true;
        }
        let Some(current) = &current.accesses else {
            self.accesses.clear();
            self.known = false;
            return;
        };
        self.accesses.extend_from_slice(current);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Pm4PhasePlan {
    indices: Vec<usize>,
    parallel: bool,
}

fn launches_are_independent(left: &RecordedHipLaunch, right: &RecordedHipLaunch) -> bool {
    let (Some(left), Some(right)) = (&left.accesses, &right.accesses) else {
        return false;
    };
    !left
        .iter()
        .any(|left| right.iter().any(|right| left.conflicts(*right)))
}

/// Partition the original HIP stream into ordered phases. Parallel phases are
/// maximal consecutive pairwise-independent antichains that meet the selected
/// width floor. Narrow antichains are folded back into the surrounding serial
/// IB so their overlap does not cost an extra cross-queue signal fan-in.
fn pm4_phase_plan(
    recorded: &[RecordedHipLaunch],
    min_parallel_width: usize,
    min_parallel_workgroups: u64,
    max_parallel_phases: usize,
) -> Vec<Pm4PhasePlan> {
    let min_parallel_width = min_parallel_width.max(2);
    let mut antichains = Vec::<Vec<usize>>::new();
    for index in 0..recorded.len() {
        let can_join = antichains.last().is_some_and(|phase| {
            phase
                .iter()
                .all(|prior| launches_are_independent(&recorded[*prior], &recorded[index]))
        });
        if can_join {
            antichains.last_mut().unwrap().push(index);
        } else {
            antichains.push(vec![index]);
        }
    }

    let mut phases = Vec::<Pm4PhasePlan>::new();
    let mut parallel_phases = 0_usize;
    for indices in antichains {
        let workgroups = indices.iter().fold(0_u64, |total, index| {
            let launch = &recorded[*index];
            let launch_workgroups = launch.grid.iter().fold(1_u64, |product, axis| {
                product.saturating_mul(u64::from(*axis))
            });
            total.saturating_add(launch_workgroups)
        });
        let parallel = indices.len() >= min_parallel_width
            && workgroups >= min_parallel_workgroups
            && parallel_phases < max_parallel_phases;
        if parallel {
            parallel_phases += 1;
        }
        if !parallel && phases.last().is_some_and(|phase| !phase.parallel) {
            phases.last_mut().unwrap().indices.extend(indices);
        } else {
            phases.push(Pm4PhasePlan { parallel, indices });
        }
    }
    phases
}

fn pm4_min_parallel_width_from_config() -> usize {
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH")
        .unwrap_or_else(|| "2".to_owned());
    value.parse::<usize>().ok().filter(|width| *width >= 2).unwrap_or_else(|| {
        eprintln!(
            "WARNING: HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH={value:?}: expected integer >= 2; using 2"
        );
        2
    })
}

fn pm4_min_parallel_workgroups_from_config() -> u64 {
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS")
        .unwrap_or_else(|| "0".to_owned());
    value.parse::<u64>().unwrap_or_else(|_| {
        eprintln!(
            "WARNING: HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS={value:?}: expected nonnegative integer; using 0"
        );
        0
    })
}

fn pm4_max_parallel_phases_from_config() -> usize {
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES")
        .unwrap_or_else(|| usize::MAX.to_string());
    value.parse::<usize>().unwrap_or_else(|_| {
        eprintln!(
            "WARNING: HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES={value:?}: expected nonnegative integer; using unlimited"
        );
        usize::MAX
    })
}

fn pm4_native_phase_sync_from_config() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_PM4_NATIVE_PHASES")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

fn pm4_queue_policy_from_config() -> QueuePolicy {
    // Keep the certified single-IB path as the default. Phase B is explicitly
    // enabled with 2, 4, or auto until hardware shadow and product gates pass.
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_QUEUES")
        .unwrap_or_else(|| "1".to_owned());
    value.parse().unwrap_or_else(|error| {
        eprintln!("WARNING: HIPFIRE_REPLAY_PM4_QUEUES={value:?}: {error}; retaining one queue");
        QueuePolicy::One
    })
}

impl ReplayTransport {
    fn from_config() -> Self {
        match hipfire_config::process_value("HIPFIRE_REPLAY_TRANSPORT")
            .unwrap_or_else(|| "aql".to_owned())
            .to_ascii_lowercase()
            .as_str()
        {
            "pm4" | "pm4_ib" | "ib" => Self::Pm4Ib,
            _ => Self::AqlPackets,
        }
    }
}

/// Experimental cache-acquire policy inside one retained PM4 tape.
///
/// The entry acquire remains unconditional: HIP populated model state and
/// kernargs before ownership crosses to the ROCr queue. `EntryOnly` removes
/// only the conservative full-system acquires between PM4 dispatches; compute
/// dependency waits and the terminal idle remain unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4MidAcquirePolicy {
    Conservative,
    EntryOnly,
    RequiredOnly,
    WithoutRepeatInterleave,
    WithoutFusedSiluRotate,
    WithoutMqRotate,
    WithoutRope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4WaitPolicy {
    Allowlist,
    ResourceAudit,
    Resource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4RegisterPolicy {
    Legacy,
    Static,
    Stateful,
}

impl Pm4RegisterPolicy {
    fn from_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "off" | "legacy" => Some(Self::Legacy),
            "static" | "static-only" | "static_only" => Some(Self::Static),
            "1" | "true" | "on" | "stateful" => Some(Self::Stateful),
            _ => None,
        }
    }

    fn from_config() -> Self {
        let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_STATEFUL")
            .unwrap_or_else(|| "static".to_owned());
        Self::from_value(&value).unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_REPLAY_PM4_STATEFUL={value:?}; \
                     retaining legacy full-register emission"
            );
            Self::Legacy
        })
    }
}

fn gfx10_dispatch_initiator_policy(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Gfx10DispatchInitiatorPolicy {
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_INITIATOR")
        .unwrap_or_else(|| "legacy".to_owned());
    let policy = gfx10_dispatch_initiator_policy_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_INITIATOR={value:?}; retaining legacy initiator"
            );
            Gfx10DispatchInitiatorPolicy::Legacy
        });
    if policy != Gfx10DispatchInitiatorPolicy::Legacy {
        eprintln!("[redline] gfx1151 PM4 dispatch initiator policy={policy:?}");
    }
    policy
}

fn gfx10_dispatch_initiator_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Gfx10DispatchInitiatorPolicy> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(Gfx10DispatchInitiatorPolicy::Legacy);
    }

    match value.to_ascii_lowercase().as_str() {
        "" | "legacy" | "ordered-append" | "ordered_append" => {
            Some(Gfx10DispatchInitiatorPolicy::Legacy)
        }
        "order" | "order-mode" | "order_mode" => Some(Gfx10DispatchInitiatorPolicy::OrderMode),
        "radv" | "order-tunnel" | "order_tunnel" => Some(Gfx10DispatchInitiatorPolicy::Radv),
        _ => None,
    }
}

fn gfx1151_dispatch_interleave(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Option<Gfx11DispatchInterleave> {
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_INTERLEAVE")
        .unwrap_or_else(|| "inherit".to_owned());
    let interleave = gfx1151_dispatch_interleave_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_INTERLEAVE={value:?}; inheriting queue value"
            );
            None
        });
    if let Some(interleave) = interleave {
        eprintln!(
            "[redline] gfx1151 PM4 dispatch interleave={} threads/SE",
            interleave.threads()
        );
    }
    interleave
}

fn gfx1151_dispatch_interleave_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Option<Gfx11DispatchInterleave>> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(None);
    }

    match value.to_ascii_lowercase().as_str() {
        "" | "inherit" | "legacy" | "firmware" => Some(None),
        "0" | "disabled" | "off" => Some(Some(Gfx11DispatchInterleave::Disabled)),
        "64" => Some(Some(Gfx11DispatchInterleave::Threads64)),
        "128" => Some(Some(Gfx11DispatchInterleave::Threads128)),
        "256" => Some(Some(Gfx11DispatchInterleave::Threads256)),
        "512" => Some(Some(Gfx11DispatchInterleave::Threads512)),
        _ => None,
    }
}

fn gfx1151_resource_limits_policy(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Gfx11ComputeResourceLimitsPolicy {
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS")
        .unwrap_or_else(|| "legacy".to_owned());
    let policy = gfx1151_resource_limits_policy_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS={value:?}; retaining zero resource limits"
            );
            Gfx11ComputeResourceLimitsPolicy::Legacy
        });
    if policy != Gfx11ComputeResourceLimitsPolicy::Legacy {
        eprintln!("[redline] gfx1151 PM4 resource-limits policy={policy:?}");
    }
    policy
}

fn gfx1151_resource_limits_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Gfx11ComputeResourceLimitsPolicy> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(Gfx11ComputeResourceLimitsPolicy::Legacy);
    }

    match value.to_ascii_lowercase().as_str() {
        "" | "legacy" | "zero" | "off" => Some(Gfx11ComputeResourceLimitsPolicy::Legacy),
        "simd-always" | "simd_always" | "always" => {
            Some(Gfx11ComputeResourceLimitsPolicy::SimdDestAlways)
        }
        // The certified gfx1151 host exposes 40 CUs over 2 SEs. Its 20 CUs/SE
        // are divisible by four, so Mesa's FORCE_SIMD_DIST guard is false.
        "radv" | "simd-dest" | "simd_dest" => Some(Gfx11ComputeResourceLimitsPolicy::Radv {
            force_simd_dist_for_single_wave: false,
        }),
        _ => None,
    }
}

fn gfx1151_cu_mask(architecture: Pm4Architecture, device_name: &str) -> Option<[u32; 2]> {
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_REDLINE_CU_COUNT")
        .unwrap_or_else(|| "all".to_owned());
    let mask = gfx1151_cu_mask_from_value(architecture, device_name, &value).unwrap_or_else(|| {
        eprintln!("WARNING: unknown HIPFIRE_GFX1151_REDLINE_CU_COUNT={value:?}; retaining all CUs");
        None
    });
    if let Some(mask) = mask {
        let enabled = mask.into_iter().map(u32::count_ones).sum::<u32>();
        eprintln!("[redline] gfx1151 queue CU mask enables {enabled}/40 CUs");
    }
    mask
}

fn gfx1151_cu_mask_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Option<[u32; 2]>> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(None);
    }
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "" | "all" | "inherit" | "40"
    ) {
        return Some(None);
    }
    let count = value.parse::<u32>().ok()?;
    if count == 0 || count >= 40 || !count.is_multiple_of(2) {
        return None;
    }
    let low = if count >= 32 {
        u32::MAX
    } else {
        (1_u32 << count) - 1
    };
    let high_count = count.saturating_sub(32);
    let high = if high_count == 0 {
        0
    } else {
        (1_u32 << high_count) - 1
    };
    Some(Some([low, high]))
}

fn gfx1151_entry_acquire_policy(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Gfx11EntryAcquirePolicy {
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE")
        .unwrap_or_else(|| "system".to_owned());
    let policy = gfx1151_entry_acquire_policy_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE={value:?}; retaining system acquire"
            );
            Gfx11EntryAcquirePolicy::System
        });
    if policy != Gfx11EntryAcquirePolicy::System {
        eprintln!("[redline] gfx1151 PM4 entry acquire policy={policy:?}");
    }
    policy
}

fn gfx1151_entry_acquire_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Gfx11EntryAcquirePolicy> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(Gfx11EntryAcquirePolicy::System);
    }
    match value.to_ascii_lowercase().as_str() {
        "" | "system" | "legacy" => Some(Gfx11EntryAcquirePolicy::System),
        "agent" | "same-agent" | "same_agent" => Some(Gfx11EntryAcquirePolicy::Agent),
        "vmem" | "vector" => Some(Gfx11EntryAcquirePolicy::Vmem),
        "none" | "raw" | "off" => Some(Gfx11EntryAcquirePolicy::None),
        _ => None,
    }
}

impl Pm4WaitPolicy {
    fn from_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "" | "allowlist" | "conservative" => Some(Self::Allowlist),
            "resource-audit" | "resource_audit" | "audit" => Some(Self::ResourceAudit),
            "resource" | "resources" => Some(Self::Resource),
            _ => None,
        }
    }

    fn from_config() -> Self {
        let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_WAIT_POLICY")
            .unwrap_or_else(|| "resource".to_owned());
        Self::from_value(&value).unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_REPLAY_PM4_WAIT_POLICY={value:?}; \
                     retaining the certified allowlist wait policy"
            );
            Self::Allowlist
        })
    }
}

#[derive(Default)]
struct Pm4WaitAudit {
    boundaries: usize,
    covered: usize,
    allowlist_independent: usize,
    resource_independent: usize,
    allowlist_only: BTreeMap<(String, String), usize>,
    resource_only: BTreeMap<(String, String), usize>,
    suballocation_candidates: BTreeMap<(String, String), usize>,
}

impl Pm4WaitAudit {
    fn observe(
        &mut self,
        previous: &RecordedHipLaunch,
        current: &RecordedHipLaunch,
        allowlist_independent: bool,
        resource_independent: bool,
        exact_start_independent: bool,
        resource_covered: bool,
    ) {
        self.boundaries += 1;
        if resource_covered {
            self.covered += 1;
        }
        self.allowlist_independent += usize::from(allowlist_independent);
        self.resource_independent += usize::from(resource_independent);
        let pair = (previous.kernel.clone(), current.kernel.clone());
        if allowlist_independent && !resource_independent {
            *self.allowlist_only.entry(pair.clone()).or_default() += 1;
        } else if resource_independent && !allowlist_independent {
            *self.resource_only.entry(pair.clone()).or_default() += 1;
        }
        if resource_covered && exact_start_independent && !resource_independent {
            *self.suballocation_candidates.entry(pair).or_default() += 1;
        }
    }

    fn report(&self, policy: Pm4WaitPolicy) {
        eprintln!(
            "[redline] PM4 wait audit policy={policy:?} boundaries={} covered={} \
             allowlist_independent={} resource_independent={} allowlist_only={:?} \
             resource_only={:?} suballocation_candidates={:?}",
            self.boundaries,
            self.covered,
            self.allowlist_independent,
            self.resource_independent,
            self.allowlist_only,
            self.resource_only,
            self.suballocation_candidates,
        );
    }
}

impl Pm4MidAcquirePolicy {
    fn from_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "" | "conservative" | "all" => Some(Self::Conservative),
            "entry-only" | "entry_only" | "none" => Some(Self::EntryOnly),
            "required-only" | "required_only" => Some(Self::RequiredOnly),
            "without-repeat-interleave" => Some(Self::WithoutRepeatInterleave),
            "without-fused-silu-rotate" => Some(Self::WithoutFusedSiluRotate),
            "without-mq-rotate" => Some(Self::WithoutMqRotate),
            "without-rope" => Some(Self::WithoutRope),
            _ => None,
        }
    }

    fn from_config() -> Self {
        let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY")
            .unwrap_or_else(|| "required-only".to_owned());
        Self::from_value(&value).unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY={value:?}; \
                 retaining conservative mid-tape acquires"
            );
            Self::Conservative
        })
    }

    fn acquire_between(self, previous: &str, current: &str) -> bool {
        match self {
            Self::Conservative => conservative_mid_acquire_except(previous, current, None),
            Self::EntryOnly => false,
            Self::RequiredOnly => required_mid_acquire(previous, current),
            Self::WithoutRepeatInterleave => {
                conservative_mid_acquire_except(previous, current, Some("repeat_interleave_qk_f32"))
            }
            Self::WithoutFusedSiluRotate => {
                conservative_mid_acquire_except(previous, current, Some("fused_silu_mul_mq_rotate"))
            }
            Self::WithoutMqRotate => {
                conservative_mid_acquire_except(previous, current, Some("mq_rotate_x"))
            }
            Self::WithoutRope => conservative_mid_acquire_except(
                previous,
                current,
                Some("rope_partial_halfsplit_f32"),
            ),
        }
    }
}

fn required_mid_acquire(previous: &str, current: &str) -> bool {
    if previous.starts_with("gated_delta_net_q8_compact2_")
        || current.starts_with("gated_delta_net_q8_compact2_")
    {
        return true;
    }
    // Dense Qwen3.5 feeds the rotated SiLU product straight into the FFN
    // down projection.  A compute-idle wait orders the dispatches, but gfx12
    // still needs the vector-cache acquire before the GEMV reads that buffer.
    // Without it the first divergent launch in the 0.8B tape is this exact
    // pair (launches 12 -> 13); logits, KV, and recurrent state then drift.
    if previous == "fused_silu_mul_mq_rotate" && current.starts_with("gemv_hfq4g256_residual") {
        return true;
    }
    matches!(
        previous,
        "repeat_interleave_qk_f32" | "rope_partial_halfsplit_f32"
    ) || matches!(
        current,
        "repeat_interleave_qk_f32" | "rope_partial_halfsplit_f32"
    )
}

fn conservative_mid_acquire_except(previous: &str, current: &str, excluded: Option<&str>) -> bool {
    if previous.starts_with("gated_delta_net_q8_compact2_")
        || current.starts_with("gated_delta_net_q8_compact2_")
    {
        return true;
    }
    (Some(previous) != excluded
        && matches!(
            previous,
            "repeat_interleave_qk_f32"
                | "fused_silu_mul_mq_rotate"
                | "mq_rotate_x"
                | "rope_partial_halfsplit_f32"
        ))
        || (Some(current) != excluded
            && matches!(
                current,
                "repeat_interleave_qk_f32"
                    | "fused_silu_mul_mq_rotate"
                    | "rope_partial_halfsplit_f32"
            ))
}

fn independent_sibling(previous: &str, current: &str) -> bool {
    matches!(
        (previous, current),
        ("fused_sigmoid_alpha_gate_f32", "conv1d_silu_split_f32")
            | ("rmsnorm_f32", "rmsnorm_f32")
            | ("kv_cache_write_q8_0", "kv_cache_write_q8_0")
            | (
                "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
                "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
            )
    )
}

impl ReplayBackendRequest {
    fn from_config() -> Self {
        match hipfire_config::process_value("HIPFIRE_REPLAY_BACKEND")
            .unwrap_or_else(|| "hip".to_owned())
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "hip" | "off" => Self::Hip,
            "shadow" => Self::Shadow,
            "auto" | "redline" => Self::Auto,
            value => {
                eprintln!("WARNING: unknown HIPFIRE_REPLAY_BACKEND={value:?}; falling back to hip");
                Self::Hip
            }
        }
    }
}

fn manual_capture_requested() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_MANUAL_CAPTURE")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

fn route_proof_log_requested() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_ROUTE_PROOF_LOG")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayState {
    Hip,
    Armed,
    RecordingWarmup,
    Captured,
    ShadowValidated,
    Ready,
    Fallback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedHipLaunch {
    pub kernel: String,
    pub artifact: Option<PathBuf>,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
    /// Optional PM4-only binding that narrows one recorded maximum grid axis
    /// from the zero-based decode position before each quiescent replay.
    pub grid_binding: Option<ReplayGridBinding>,
    /// Exact naturally-aligned, tail-padded bytes passed through HIP's
    /// contiguous `extra` launch ABI. The model adapter owns the lifetime
    /// contract for pointer values recovered into allocation-wide effects.
    pub kernarg: Vec<u8>,
    /// Allocation-wide effects recovered from typed kernel signatures and
    /// `hipMemGetAddressRange`. `None` means the launch must remain serialized.
    accesses: Option<Vec<RecordedResourceAccess>>,
}

/// A dynamic retained-grid contract supplied by the engine at capture time.
/// The recorded grid remains the hard maximum; replay may only narrow it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayGridBinding {
    PositionCeilDiv { axis: u8, addend: u32, divisor: u32 },
}

impl ReplayGridBinding {
    fn bind(self, position: usize, recorded: [u32; 3]) -> Result<[u32; 3], String> {
        let Self::PositionCeilDiv {
            axis,
            addend,
            divisor,
        } = self;
        let axis = usize::from(axis);
        if axis >= 3 || divisor == 0 {
            return Err(format!(
                "invalid replay grid binding axis={axis} divisor={divisor}"
            ));
        }
        let extent = u64::try_from(position)
            .map_err(|_| "decode position exceeds u64".to_owned())?
            .checked_add(u64::from(addend))
            .ok_or_else(|| "dynamic replay extent overflow".to_owned())?;
        let units = extent.div_ceil(u64::from(divisor)).max(1);
        let units =
            u32::try_from(units).map_err(|_| format!("dynamic replay grid {units} exceeds u32"))?;
        let mut bound = recorded;
        bound[axis] = units.min(recorded[axis]);
        Ok(bound)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCaptureSummary {
    pub launch_count: usize,
    pub unique_kernel_count: usize,
    pub sequence_hash: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayObservation {
    pub count: u64,
    pub first_position: Option<usize>,
    pub last_position: Option<usize>,
    pub failed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedReplayIdentity {
    pub dispatch_count: usize,
    pub packet_count: Option<usize>,
    pub queue_id: u64,
    pub command_dwords: Option<u32>,
    /// Prepared graph queue width (AQL linear batch is always 1).
    pub queue_count: usize,
    /// Prepared graph logical phase count (AQL linear batch is always 1).
    pub phase_count: usize,
}

fn pm4_packet_identity(queue_count: usize, phase_count: usize) -> Option<usize> {
    (queue_count == 1 && phase_count == 1).then_some(1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AqlContractProbe {
    pub kernel: String,
    pub captured_kernarg_bytes: usize,
    pub loader_kernarg_bytes: u32,
    pub loader_kernarg_alignment: u32,
    pub static_group_bytes: u32,
    pub dynamic_group_bytes: u32,
}

pub struct PreparedLinearAqlReplay {
    graph: SingleQueueBatchGraph,
    dynamic_gdn_frames: Vec<usize>,
}

impl PreparedLinearAqlReplay {
    /// # Safety
    ///
    /// Every pointer captured in the immutable explicit kernarg prefixes must
    /// still refer to the same live Hipfire allocation and model instance.
    pub unsafe fn replay_and_wait(&mut self) -> Result<GpuBatchTiming, String> {
        for dispatch in &self.dynamic_gdn_frames {
            let frame = crate::norm::reserve_gdn_requant_frames(1);
            self.graph
                .patch_kernarg_u32(*dispatch, 76, frame)
                .map_err(|error| error.to_string())?;
        }
        // SAFETY: forwarded from the caller that owns the model allocations.
        unsafe { self.graph.replay_and_wait() }.map_err(|error| error.to_string())
    }

    pub fn dispatch_count(&self) -> usize {
        self.graph.dispatch_count()
    }

    pub fn packet_count(&self) -> usize {
        self.graph.packet_count()
    }

    pub fn queue_id(&self) -> u64 {
        self.graph.queue_id()
    }
}

/// True when the per-dispatch timestamp diagnostic is requested.
pub fn dispatch_profile_enabled() -> bool {
    hipfire_config::developer_var_os("HIPFIRE_REDLINE_DISPATCH_PROFILE")
        .is_some_and(|value| value != "0" && !value.is_empty())
}

/// Summarise per-dispatch spans so a slow machine reports a distribution
/// rather than a single throughput number.
///
/// The shape is the diagnostic: overhead spread evenly across every dispatch
/// points at per-dispatch cost (fetch, launch, submission), whereas a few
/// dispatches dominating points at specific stalls — a barrier or a cache
/// release. Those two have different causes and different fixes, and an
/// end-to-end tok/s cannot tell them apart.
fn report_dispatch_spans(spans: &[u64]) {
    if spans.is_empty() {
        return;
    }
    let mut sorted: Vec<u64> = spans.to_vec();
    sorted.sort_unstable();
    let total: u64 = sorted.iter().sum();
    let pick = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    // Contribution of the slowest 5% — the number that separates "everything is
    // slightly slow" from "a handful of dispatches dominate".
    let tail_start = sorted.len() - sorted.len().div_ceil(20);
    let tail: u64 = sorted[tail_start..].iter().sum();
    eprintln!(
        "[redline] dispatch spans n={} total={}us p50={}ns p90={}ns p99={}ns max={}ns \
         slowest5%={:.1}% of total",
        sorted.len(),
        total / 1_000,
        pick(0.50),
        pick(0.90),
        pick(0.99),
        sorted[sorted.len() - 1],
        100.0 * tail as f64 / total.max(1) as f64,
    );
}

enum PreparedPm4Graph {
    Single(SingleQueuePm4Ib),
    Phased(PhasedMultiQueuePm4Ib),
}

impl PreparedPm4Graph {
    unsafe fn replay_and_wait_profiled(&mut self) -> Result<GpuMultiQueueTiming, String> {
        if dispatch_profile_enabled() {
            if let Self::Single(graph) = self {
                // Execute the instrumented graph once. Reuse the same timestamp
                // vector for whole-tape timing and the one-line legacy report.
                let (timing, spans) = unsafe { graph.replay_and_wait_dispatch_profiled() }
                    .map_err(|error| error.to_string())?;
                static REPORTED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    report_dispatch_spans(&spans);
                }
                return Ok(timing);
            }
        }
        match self {
            Self::Single(graph) => unsafe { graph.replay_and_wait_profiled() },
            Self::Phased(graph) => unsafe { graph.replay_and_wait_profiled() },
        }
        .map_err(|error| error.to_string())
    }

    fn read_dispatch_profile(&mut self) -> Result<Pm4DispatchProfile, String> {
        match self {
            Self::Single(graph) => graph
                .read_dispatch_profile()
                .map(|(timing, spans_nanoseconds)| Pm4DispatchProfile {
                    timing,
                    spans_nanoseconds,
                })
                .map_err(|error| error.to_string()),
            Self::Phased(_) => {
                Err("per-dispatch PM4 profiling requires a single retained queue".to_owned())
            }
        }
    }

    fn queue_id(&self) -> u64 {
        match self {
            Self::Single(graph) => graph.queue_id(),
            Self::Phased(graph) => graph
                .queue_ids()
                .next()
                .expect("prepared phased PM4 replay owns at least one queue"),
        }
    }

    fn queue_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Phased(graph) => graph.queue_count(),
        }
    }

    fn phase_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Phased(graph) => graph.phase_count(),
        }
    }

    fn patch_dispatch_dimensions(
        &mut self,
        dispatch: usize,
        dimensions: [u32; 3],
    ) -> Result<(), String> {
        match self {
            Self::Single(graph) => graph
                .patch_dispatch_dimensions(dispatch, dimensions)
                .map_err(|error| error.to_string()),
            Self::Phased(_) => {
                Err("dynamic PM4 geometry requires the certified single-queue replay".to_owned())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Pm4DispatchBoundary {
    /// Entry ownership acquire emitted before the first dispatch.
    /// Distinct from mid-tape `acquire_inter_node`.
    pub entry_acquire: bool,
    pub wait_compute_idle: bool,
    pub acquire_inter_node: bool,
    pub acquire_vmem: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pm4DispatchProfile {
    pub timing: GpuMultiQueueTiming,
    pub spans_nanoseconds: Vec<u64>,
}

pub struct PreparedPm4Replay {
    graph: PreparedPm4Graph,
    // Kernels retain their HSA executables and kernargs retain every pointer
    // programmed into the immutable indirect buffer.
    _kernels: Vec<Kernel>,
    kernargs: Vec<KernargBuffer>,
    dynamic_gdn_frames: Vec<usize>,
    dynamic_grids: Vec<(usize, ReplayGridBinding, [u32; 3], [u32; 3])>,
    pm4_architecture: Pm4Architecture,
    dispatch_count: usize,
    command_dwords: u32,
    dispatch_boundaries: Option<Vec<Pm4DispatchBoundary>>,
}

impl PreparedPm4Replay {
    /// # Safety
    ///
    /// Every pointer captured in the immutable explicit kernarg prefixes must
    /// still refer to the same live Hipfire allocation and model instance.
    pub unsafe fn replay_and_wait(
        &mut self,
        position: usize,
    ) -> Result<GpuMultiQueueTiming, String> {
        for dispatch in &self.dynamic_gdn_frames {
            let frame = crate::norm::reserve_gdn_requant_frames(1);
            let bytes = self.kernargs[*dispatch].as_mut_bytes();
            bytes
                .get_mut(76..80)
                .ok_or_else(|| "PM4 GDN kernarg is too short for frame patch".to_owned())?
                .copy_from_slice(&frame.to_ne_bytes());
        }
        for (dispatch, binding, recorded, workgroup) in &self.dynamic_grids {
            let workgroups = binding.bind(position, *recorded)?;
            let dimensions = if self.pm4_architecture == Pm4Architecture::Gfx12 {
                let mut workitems = [0_u32; 3];
                for axis in 0..3 {
                    workitems[axis] =
                        workgroups[axis]
                            .checked_mul(workgroup[axis])
                            .ok_or_else(|| {
                                format!(
                                "dynamic PM4 grid overflow axis={axis} workgroups={} workgroup={}",
                                workgroups[axis], workgroup[axis]
                            )
                            })?;
                }
                workitems
            } else {
                workgroups
            };
            self.graph
                .patch_dispatch_dimensions(*dispatch, dimensions)?;
        }
        // SAFETY: forwarded from the caller that owns the model allocations.
        unsafe { self.graph.replay_and_wait_profiled() }
    }

    /// Replay one instrumented retained graph exactly once.
    ///
    /// # Safety
    ///
    /// Every captured pointer must remain live, as for [`Self::replay_and_wait`].
    pub unsafe fn replay_and_wait_dispatch_profiled(
        &mut self,
        position: usize,
    ) -> Result<Pm4DispatchProfile, String> {
        if self.dispatch_boundaries.is_none() {
            return Err("prepared PM4 graph has no per-dispatch timestamps".to_owned());
        }
        // SAFETY: forwarded from the caller that owns the model allocations.
        unsafe { self.replay_and_wait(position)? };
        self.graph.read_dispatch_profile()
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    pub fn command_dwords(&self) -> u32 {
        self.command_dwords
    }

    pub fn dispatch_boundaries(&self) -> Option<&[Pm4DispatchBoundary]> {
        self.dispatch_boundaries.as_deref()
    }

    pub fn queue_id(&self) -> u64 {
        self.graph.queue_id()
    }

    pub fn queue_count(&self) -> usize {
        self.graph.queue_count()
    }

    pub fn phase_count(&self) -> usize {
        self.graph.phase_count()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowValidation {
    pub bit_exact: bool,
    pub guards_intact: bool,
    pub same_artifact: bool,
    pub abi_valid: bool,
    pub automatic_clocks: bool,
    pub gpu_timed: bool,
    pub speedup_over_hip: f64,
}

impl ShadowValidation {
    fn passes(self, threshold: f64) -> bool {
        self.bit_exact
            && self.guards_intact
            && self.same_artifact
            && self.abi_valid
            && self.automatic_clocks
            && self.gpu_timed
            && self.speedup_over_hip.is_finite()
            && self.speedup_over_hip >= threshold
    }
}

/// Process-local replay adoption state. HIP remains the route until an adapter
/// both supplies two certified observations and installs a concrete prepared
/// plan. Any failure permanently falls back for this controller.
pub struct ReplayController {
    request: ReplayBackendRequest,
    transport: ReplayTransport,
    pm4_mid_acquire_policy: Pm4MidAcquirePolicy,
    pm4_wait_policy: Pm4WaitPolicy,
    pm4_register_policy: Pm4RegisterPolicy,
    pm4_queue_policy: QueuePolicy,
    state: ReplayState,
    recorded: Vec<RecordedHipLaunch>,
    certified_speedups: Vec<f64>,
    threshold: f64,
    max_recorded_launches: usize,
    fallback_reason: Option<String>,
    prepared: Option<PreparedLinearAqlReplay>,
    prepared_pm4: Option<PreparedPm4Replay>,
    auto_lifecycle: bool,
    forward_eligible: bool,
    replay_observation: ReplayObservation,
    /// Opt-in latch for daemon-owned post-generate route-proof markers.
    route_proof_log: bool,
}

impl ReplayController {
    pub fn from_config() -> Self {
        let request = ReplayBackendRequest::from_config();
        let manual = manual_capture_requested();
        let mut controller = if manual {
            Self::new_armed(request)
        } else {
            Self::new(request)
        };
        controller.auto_lifecycle = !manual;
        if !manual && request != ReplayBackendRequest::Hip {
            // Model load and prefill priming use the same central launch
            // recorder. Arm here and clear/start only at the first eligible
            // plain-AR forward so the retained tape cannot absorb setup work.
            controller.state = ReplayState::Armed;
        }
        controller
    }

    pub fn new(request: ReplayBackendRequest) -> Self {
        let state = if request == ReplayBackendRequest::Hip {
            ReplayState::Hip
        } else {
            ReplayState::RecordingWarmup
        };
        Self {
            request,
            transport: ReplayTransport::from_config(),
            pm4_mid_acquire_policy: Pm4MidAcquirePolicy::from_config(),
            pm4_wait_policy: Pm4WaitPolicy::from_config(),
            pm4_register_policy: Pm4RegisterPolicy::from_config(),
            pm4_queue_policy: pm4_queue_policy_from_config(),
            state,
            recorded: Vec::new(),
            certified_speedups: Vec::new(),
            threshold: 1.03,
            max_recorded_launches: 4096,
            fallback_reason: None,
            prepared: None,
            prepared_pm4: None,
            auto_lifecycle: false,
            forward_eligible: true,
            replay_observation: ReplayObservation::default(),
            route_proof_log: route_proof_log_requested(),
        }
    }

    pub fn new_armed(request: ReplayBackendRequest) -> Self {
        let mut controller = Self::new(request);
        if request != ReplayBackendRequest::Hip {
            controller.state = ReplayState::Armed;
        }
        controller
    }

    /// Apply the daemon's model-scoped replay default after a successful load.
    ///
    /// An explicit backend selection always wins. Otherwise every successful
    /// model load resets the process-local controller so prepared queues,
    /// command buffers, and fallback state cannot bleed across model swaps.
    /// The exact A3B MQ4R runtime predicate may default to retained PM4 on
    /// gfx1100, gfx1151, and gfx1201; this is runtime policy, not certification.
    /// All other models return to ordinary HIP. An explicit transport still
    /// overrides the PM4 transport choice for diagnostics.
    pub fn configure_model_default(&mut self, enable_a3b_mq4r: bool) -> bool {
        let manual = manual_capture_requested();
        let backend = hipfire_config::process_value("HIPFIRE_REPLAY_BACKEND");
        let explicit_backend = backend.as_deref().is_some_and(|value| value != "auto");
        if explicit_backend || manual {
            self.reset_for_model(
                ReplayBackendRequest::from_config(),
                ReplayTransport::from_config(),
                !manual,
            );
            return false;
        }

        let transport_override = hipfire_config::process_value("HIPFIRE_REPLAY_TRANSPORT");
        let transport = if enable_a3b_mq4r
            && !transport_override
                .as_deref()
                .is_some_and(|value| value != "auto")
        {
            ReplayTransport::Pm4Ib
        } else {
            ReplayTransport::from_config()
        };
        self.apply_model_default(enable_a3b_mq4r, transport);
        true
    }

    fn apply_model_default(&mut self, enable_a3b_mq4r: bool, transport: ReplayTransport) {
        let request = if enable_a3b_mq4r {
            ReplayBackendRequest::Auto
        } else {
            ReplayBackendRequest::Hip
        };
        self.reset_for_model(request, transport, true);
    }

    fn reset_for_model(
        &mut self,
        request: ReplayBackendRequest,
        transport: ReplayTransport,
        auto_lifecycle: bool,
    ) {
        self.request = request;
        self.transport = transport;
        self.state = if request == ReplayBackendRequest::Hip {
            ReplayState::Hip
        } else {
            ReplayState::Armed
        };
        self.recorded.clear();
        self.certified_speedups.clear();
        self.fallback_reason = None;
        self.prepared = None;
        self.prepared_pm4 = None;
        self.auto_lifecycle = auto_lifecycle;
        self.forward_eligible = true;
        self.replay_observation = ReplayObservation::default();
    }

    pub fn transport_name(&self) -> &'static str {
        match self.transport {
            ReplayTransport::AqlPackets => "aql",
            ReplayTransport::Pm4Ib => "pm4",
        }
    }

    pub fn request(&self) -> ReplayBackendRequest {
        self.request
    }

    pub fn state(&self) -> ReplayState {
        self.state
    }

    pub fn recorded_launches(&self) -> &[RecordedHipLaunch] {
        &self.recorded
    }

    pub fn pm4_queue_policy(&self) -> QueuePolicy {
        self.pm4_queue_policy
    }

    pub fn prepared_pm4_shape(&self) -> Option<(usize, usize)> {
        self.prepared_pm4
            .as_ref()
            .map(|prepared| (prepared.queue_count(), prepared.phase_count()))
    }

    pub fn prepared_route_identity(&self) -> Option<PreparedReplayIdentity> {
        match self.transport {
            ReplayTransport::AqlPackets => {
                self.prepared
                    .as_ref()
                    .map(|prepared| PreparedReplayIdentity {
                        dispatch_count: prepared.dispatch_count(),
                        packet_count: Some(prepared.packet_count()),
                        queue_id: prepared.queue_id(),
                        command_dwords: None,
                        // Linear AQL is a single-queue, single-phase batch graph.
                        queue_count: 1,
                        phase_count: 1,
                    })
            }
            ReplayTransport::Pm4Ib => {
                self.prepared_pm4
                    .as_ref()
                    .map(|prepared| PreparedReplayIdentity {
                        dispatch_count: prepared.dispatch_count(),
                        packet_count: pm4_packet_identity(
                            prepared.queue_count(),
                            prepared.phase_count(),
                        ),
                        queue_id: prepared.queue_id(),
                        command_dwords: Some(prepared.command_dwords()),
                        queue_count: prepared.queue_count(),
                        phase_count: prepared.phase_count(),
                    })
            }
        }
    }

    pub fn replay_observation(&self) -> ReplayObservation {
        self.replay_observation
    }

    /// Start a request-local route-proof window without changing replay state.
    pub fn begin_replay_observation_window(&mut self) {
        self.replay_observation = ReplayObservation::default();
    }

    /// Invalidate the current request-local proof window after cancellation or
    /// another request-level failure that is not itself a replay error.
    pub fn invalidate_replay_observation_window(&mut self) {
        self.replay_observation.failed = true;
    }

    /// Build one request-scoped retained-replay proof marker.
    ///
    /// The daemon owns request boundaries and stderr emission. Invalid request
    /// IDs fail closed so an untrusted ID cannot inject or alias log lines.
    pub fn replay_observation_marker(&self, request_id: &str) -> Option<String> {
        if !self.route_proof_log
            || self.replay_observation.failed
            || self.replay_observation.count == 0
            || request_id.is_empty()
            || !request_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
            })
        {
            return None;
        }
        let position = self.replay_observation.first_position?;
        Some(format!(
            "HIPFIRE_REPLAY_ROUTE_PROOF transport={} position={} request_id={} replays={}",
            self.transport_name(),
            position,
            request_id,
            self.replay_observation.count
        ))
    }

    pub fn is_recording(&self) -> bool {
        self.state == ReplayState::RecordingWarmup && self.forward_eligible
    }

    /// Apply the model's one-shot plain-AR eligibility decision to this
    /// forward. Speculative/MTP re-seed and verify calls must neither populate
    /// the plain-AR capture nor route its prepared replay.
    pub fn set_forward_eligible(&mut self, eligible: bool) {
        self.forward_eligible = eligible;
    }

    pub fn is_enabled(&self) -> bool {
        self.request != ReplayBackendRequest::Hip && self.state != ReplayState::Fallback
    }

    pub fn should_auto_finalize_capture(&self) -> bool {
        self.auto_lifecycle && self.is_recording()
    }

    pub fn begin_auto_capture_if_armed(&mut self) -> Result<(), &'static str> {
        if self.auto_lifecycle && self.forward_eligible && self.state == ReplayState::Armed {
            self.begin_capture()?;
        }
        Ok(())
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    /// Load every distinct captured HIP artifact through public HSA and prove
    /// that its loader-reported kernarg ABI accepts the exact padded bytes the
    /// HIP launch used. This creates no queue and executes no packet.
    pub fn probe_aql_contracts(
        &self,
        device_ordinal: usize,
    ) -> Result<Vec<AqlContractProbe>, String> {
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(device_ordinal))
            .map_err(|error| error.to_string())?;
        let mut seen = BTreeSet::new();
        let mut probes = Vec::new();
        for launch in &self.recorded {
            if !seen.insert(launch.kernel.clone()) {
                continue;
            }
            let artifact = launch.artifact.as_ref().ok_or_else(|| {
                format!("captured kernel {:?} has no owning HSACO", launch.kernel)
            })?;
            let bytes: Arc<[u8]> = std::fs::read(artifact)
                .map_err(|error| format!("read {}: {error}", artifact.display()))?
                .into();
            let executable = Executable::load(&device, bytes)
                .map_err(|error| format!("load {}: {error}", artifact.display()))?;
            let symbol = format!("{}.kd", launch.kernel);
            let kernel = executable
                .kernel(&symbol)
                .map_err(|error| format!("resolve {symbol}: {error}"))?;
            let metadata = kernel.metadata();
            validate_loader_kernarg(launch, metadata.kernarg_segment_size as usize)
                .map_err(|reason| format!("{symbol}: {reason}"))?;
            probes.push(AqlContractProbe {
                kernel: launch.kernel.clone(),
                captured_kernarg_bytes: launch.kernarg.len(),
                loader_kernarg_bytes: metadata.kernarg_segment_size,
                loader_kernarg_alignment: metadata.kernarg_segment_alignment,
                static_group_bytes: metadata.group_segment_size,
                dynamic_group_bytes: launch.shared_mem,
            });
        }
        Ok(probes)
    }

    /// Lower the exact captured HIP sequence to one public-HSA queue. All
    /// explicit argument bytes remain unchanged; only the standardized
    /// 256-byte gfx12 implicit-argument suffix is synthesized from launch
    /// geometry, matching CLR's module-launch path.
    pub fn prepare_linear_aql(
        &mut self,
        device_ordinal: usize,
    ) -> Result<(usize, usize, u64), String> {
        self.prepare_linear_aql_prefix(device_ordinal, self.recorded.len())
    }

    pub fn prepare_linear_aql_prefix(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, usize, u64), String> {
        if self.recorded.is_empty() {
            return Err("no captured launch sequence".to_owned());
        }
        if prefix < 2 || prefix > self.recorded.len() {
            return Err(format!(
                "AQL prefix {prefix} must be in 2..={}",
                self.recorded.len()
            ));
        }
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(device_ordinal))
            .map_err(|error| error.to_string())?;
        let pool = KernargPool::discover(&device).map_err(|error| error.to_string())?;
        let mut executables = BTreeMap::<PathBuf, Executable>::new();
        let mut kernels = BTreeMap::<(PathBuf, String), Kernel>::new();
        let mut dispatches = Vec::with_capacity(prefix);
        let mut dynamic_gdn_frames = Vec::new();

        for launch in self.recorded.iter().take(prefix) {
            let artifact = launch.artifact.clone().ok_or_else(|| {
                format!("captured kernel {:?} has no owning HSACO", launch.kernel)
            })?;
            if !executables.contains_key(&artifact) {
                let bytes: Arc<[u8]> = std::fs::read(&artifact)
                    .map_err(|error| format!("read {}: {error}", artifact.display()))?
                    .into();
                let executable = Executable::load(&device, bytes)
                    .map_err(|error| format!("load {}: {error}", artifact.display()))?;
                executables.insert(artifact.clone(), executable);
            }
            let symbol = format!("{}.kd", launch.kernel);
            let key = (artifact.clone(), symbol.clone());
            if !kernels.contains_key(&key) {
                let kernel = executables[&artifact]
                    .kernel(&symbol)
                    .map_err(|error| format!("resolve {symbol}: {error}"))?;
                kernels.insert(key.clone(), kernel);
            }
            let kernel = kernels[&key].clone();
            let metadata = kernel.metadata();
            let mut kernarg = pool
                .allocate_for(metadata)
                .map_err(|error| format!("allocate {symbol} kernarg: {error}"))?;
            populate_gfx12_kernarg(&mut kernarg, launch, metadata.kernarg_segment_size as usize)?;
            let mut workgroup = [0_u16; 3];
            for (axis, value) in launch.block.into_iter().enumerate() {
                workgroup[axis] = u16::try_from(value)
                    .map_err(|_| format!("{symbol}: workgroup dimension {value} exceeds u16"))?;
            }
            let geometry = LaunchGeometry::from_hip_workgroups(launch.grid, workgroup)
                .map_err(|error| format!("{symbol}: {error}"))?;
            let dispatch = RecordedDispatch::new(0, kernel, geometry, kernarg)
                .map_err(|error| format!("{symbol}: {error}"))?
                .with_dynamic_group_bytes(launch.shared_mem)
                .map_err(|error| format!("{symbol}: {error}"))?;
            if launch.kernel == "gated_delta_net_q8_fast"
                || launch.kernel.starts_with("gated_delta_net_q8_compact2_")
            {
                if metadata.kernarg_segment_size < 80 {
                    return Err(format!(
                        "{symbol}: loader kernarg is too short for dynamic frame binding"
                    ));
                }
                dynamic_gdn_frames.push(dispatches.len());
            }
            dispatches.push(dispatch);
        }

        let required = dispatches
            .len()
            .checked_add(1)
            .ok_or_else(|| "AQL packet count overflow".to_owned())?;
        let queue_size = required
            .next_power_of_two()
            .max(*device.queue_size_range().start() as usize);
        let queue_size = u32::try_from(queue_size)
            .map_err(|_| format!("AQL queue size {queue_size} exceeds u32"))?;
        if !device.queue_size_range().contains(&queue_size) {
            return Err(format!(
                "AQL queue size {queue_size} outside {:?}",
                device.queue_size_range()
            ));
        }
        let mut headers = vec![HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL; dispatches.len()];
        headers[0] = HeaderPolicy::BATCH_BOUNDARY_FIRST_SERIAL;
        for (index, launch) in self.recorded.iter().take(prefix).enumerate() {
            if launch.kernel == "repeat_interleave_qk_f32" {
                headers[index] = HeaderPolicy::RECORDED_DISPATCH;
                if index + 1 < headers.len() {
                    headers[index + 1] = HeaderPolicy::BATCH_INTERNAL_ACQUIRE_SYSTEM;
                }
            } else if matches!(
                launch.kernel.as_str(),
                "fused_silu_mul_mq_rotate" | "mq_rotate_x" | "rope_partial_halfsplit_f32"
            ) {
                headers[index] = if launch.kernel == "mq_rotate_x" {
                    HeaderPolicy::BATCH_INTERNAL_RELEASE_SYSTEM
                } else {
                    HeaderPolicy::RECORDED_DISPATCH
                };
            }
        }
        for index in 1..headers.len() {
            let previous = self.recorded[index - 1].kernel.as_str();
            let current = self.recorded[index].kernel.as_str();
            if independent_sibling(previous, current) {
                headers[index] = HeaderPolicy::BATCH_BOUNDARY_INTERNAL_INDEPENDENT;
            }
        }
        let graph = if self.request == ReplayBackendRequest::Auto {
            SingleQueueBatchGraph::create_unprofiled_with_dispatch_headers(
                &device,
                queue_size,
                dispatches,
                BatchFencePolicy::BoundarySerialized,
                headers,
            )
        } else {
            SingleQueueBatchGraph::create_with_dispatch_headers(
                &device,
                queue_size,
                dispatches,
                BatchFencePolicy::BoundarySerialized,
                headers,
            )
        }
        .map_err(|error| error.to_string())?;
        let summary = (
            graph.dispatch_count(),
            graph.packet_count(),
            graph.queue_id(),
        );
        self.prepared = Some(PreparedLinearAqlReplay {
            graph,
            dynamic_gdn_frames,
        });
        self.state = ReplayState::Ready;
        Ok(summary)
    }

    /// Lower a captured prefix to one retained architecture-native PM4
    /// indirect buffer. Unsupported HSA agents fail closed before commands are
    /// constructed; gfx10/11 and gfx12 never share register encodings.
    pub fn prepare_pm4_prefix(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, false)
    }

    /// Lower a captured prefix to a single GFX12 IB with per-dispatch timestamps.
    pub fn prepare_pm4_dispatch_profile(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, true)
    }

    fn prepare_pm4_prefix_inner(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
        dispatch_profile: bool,
    ) -> Result<(usize, u32, u64), String> {
        if self.recorded.is_empty() {
            return Err("no captured launch sequence".to_owned());
        }
        if prefix == 0 || prefix > self.recorded.len() {
            return Err(format!(
                "PM4 prefix {prefix} must be in 1..={}",
                self.recorded.len()
            ));
        }
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(device_ordinal))
            .map_err(|error| error.to_string())?;
        let pm4_architecture = Pm4Architecture::from_device(&device)?;
        if dispatch_profile && pm4_architecture != Pm4Architecture::Gfx12 {
            return Err("per-dispatch PM4 profiling currently requires gfx12".to_owned());
        }
        let dispatch_initiator_policy =
            gfx10_dispatch_initiator_policy(pm4_architecture, device.name());
        let dispatch_interleave = gfx1151_dispatch_interleave(pm4_architecture, device.name());
        let resource_limits_policy =
            gfx1151_resource_limits_policy(pm4_architecture, device.name());
        let cu_mask = gfx1151_cu_mask(pm4_architecture, device.name());
        let entry_acquire_policy = gfx1151_entry_acquire_policy(pm4_architecture, device.name());
        let pool = KernargPool::discover(&device).map_err(|error| error.to_string())?;
        let mut executables = BTreeMap::<PathBuf, Executable>::new();
        let mut resolved = BTreeMap::<(PathBuf, String), Kernel>::new();
        let mut kernels = Vec::with_capacity(prefix);
        let mut kernargs = Vec::with_capacity(prefix);
        let mut geometries = Vec::with_capacity(prefix);
        let mut dynamic_gdn_frames = Vec::new();
        let mut dynamic_grids = Vec::new();

        for launch in self.recorded.iter().take(prefix) {
            let artifact = launch.artifact.clone().ok_or_else(|| {
                format!("captured kernel {:?} has no owning HSACO", launch.kernel)
            })?;
            if !executables.contains_key(&artifact) {
                let bytes: Arc<[u8]> = std::fs::read(&artifact)
                    .map_err(|error| format!("read {}: {error}", artifact.display()))?
                    .into();
                let executable = Executable::load(&device, bytes)
                    .map_err(|error| format!("load {}: {error}", artifact.display()))?;
                executables.insert(artifact.clone(), executable);
            }
            let symbol = format!("{}.kd", launch.kernel);
            let key = (artifact.clone(), symbol.clone());
            if !resolved.contains_key(&key) {
                let kernel = executables[&artifact]
                    .kernel(&symbol)
                    .map_err(|error| format!("resolve {symbol}: {error}"))?;
                resolved.insert(key.clone(), kernel);
            }
            let kernel = resolved[&key].clone();
            let metadata = kernel.metadata();
            let mut kernarg = pool
                .allocate_for(metadata)
                .map_err(|error| format!("allocate {symbol} kernarg: {error}"))?;
            populate_gfx12_kernarg(&mut kernarg, launch, metadata.kernarg_segment_size as usize)?;
            let mut workgroup = [0_u16; 3];
            for (axis, value) in launch.block.into_iter().enumerate() {
                workgroup[axis] = u16::try_from(value)
                    .map_err(|_| format!("{symbol}: workgroup dimension {value} exceeds u16"))?;
            }
            let geometry = LaunchGeometry::from_hip_workgroups(launch.grid, workgroup)
                .map_err(|error| format!("{symbol}: {error}"))?;
            device
                .validate_geometry(geometry)
                .map_err(|error| format!("{symbol}: {error}"))?;
            if launch.kernel == "gated_delta_net_q8_fast"
                || launch.kernel.starts_with("gated_delta_net_q8_compact2_")
            {
                if metadata.kernarg_segment_size < 80 {
                    return Err(format!(
                        "{symbol}: loader kernarg is too short for dynamic frame binding"
                    ));
                }
                dynamic_gdn_frames.push(kernargs.len());
            }
            if let Some(binding) = launch.grid_binding {
                dynamic_grids.push((kernargs.len(), binding, launch.grid, launch.block));
            }
            kernels.push(kernel);
            kernargs.push(kernarg);
            geometries.push(geometry);
        }

        let gfx12_gcr_trim = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GCR_TRIM")
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or(true);
        let gfx11_vmem_acquire =
            match hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE") {
                Some(value) if value != "auto" => matches!(value.as_str(), "1" | "true" | "on"),
                _ => device.name().eq_ignore_ascii_case("gfx1151"),
            };
        let mut wait_audit = Pm4WaitAudit::default();
        let mut audit_frontier = ResourceFrontier::default();
        for index in 0..prefix {
            if index != 0 {
                let previous_launch = &self.recorded[index - 1];
                let current_launch = &self.recorded[index];
                let previous = previous_launch.kernel.as_str();
                let current = current_launch.kernel.as_str();
                let allowlist_independent = independent_sibling(previous, current);
                let resource_covered = audit_frontier.covered(current_launch);
                let resources_independent = audit_frontier.independent(current_launch);
                let exact_start_independent =
                    audit_frontier.independent_by_exact_start(current_launch);
                wait_audit.observe(
                    previous_launch,
                    current_launch,
                    allowlist_independent,
                    resources_independent,
                    exact_start_independent,
                    resource_covered,
                );
                audit_frontier.advance(current_launch, resources_independent);
            } else {
                audit_frontier.advance(&self.recorded[index], false);
            }
        }
        if self.pm4_wait_policy != Pm4WaitPolicy::Allowlist {
            wait_audit.report(self.pm4_wait_policy);
        }

        // Dynamic direct-dispatch patching is intentionally limited to the
        // certified one-IB path. Multi-queue phases duplicate and reorder
        // command buffers, so a global capture index is not a safe patch key.
        let queue_limit = if dispatch_profile {
            1
        } else if dynamic_grids.is_empty() {
            self.pm4_queue_policy.resolve(device.name(), usize::MAX)
        } else {
            1
        };
        if cu_mask.is_some() && queue_limit != 1 {
            return Err("gfx1151 CU-mask experiments require single-queue PM4 replay".to_owned());
        }
        let mut dispatch_boundaries = Vec::new();
        let (graph, command_dwords) = if queue_limit == 1 {
            let mut commands = Pm4Commands::new(
                pm4_architecture,
                self.pm4_register_policy,
                dispatch_initiator_policy,
                dispatch_interleave,
                resource_limits_policy,
            );
            commands.acquire_entry(gfx12_gcr_trim, entry_acquire_policy);
            let mut resource_frontier = ResourceFrontier::default();
            for index in 0..prefix {
                let mut boundary = Pm4DispatchBoundary::default();
                if index != 0 {
                    let previous_launch = &self.recorded[index - 1];
                    let current_launch = &self.recorded[index];
                    let previous = previous_launch.kernel.as_str();
                    let current = current_launch.kernel.as_str();
                    let allowlist_independent = independent_sibling(previous, current);
                    let resources_independent = resource_frontier.independent(current_launch);
                    let independent = match self.pm4_wait_policy {
                        Pm4WaitPolicy::Allowlist | Pm4WaitPolicy::ResourceAudit => {
                            allowlist_independent
                        }
                        Pm4WaitPolicy::Resource => resources_independent,
                    };
                    if !independent {
                        commands.wait_compute_idle();
                    }
                    resource_frontier.advance(current_launch, resources_independent);
                    if (!independent && commands.requires_dependency_acquire())
                        || self
                            .pm4_mid_acquire_policy
                            .acquire_between(previous, current)
                    {
                        boundary.acquire_vmem =
                            pm4_vmem_acquire_enabled(pm4_architecture, gfx11_vmem_acquire, current);
                        commands.acquire_inter_node(gfx12_gcr_trim, boundary.acquire_vmem);
                    }
                } else {
                    resource_frontier.advance(&self.recorded[index], false);
                }
                commands
                    .dispatch(
                        &kernels[index],
                        geometries[index],
                        self.recorded[index].shared_mem,
                        kernargs[index].address(),
                    )
                    .map_err(|error| format!("{}: {error}", self.recorded[index].kernel))?;
                if dispatch_profile {
                    dispatch_boundaries.push(boundary);
                }
            }
            commands.wait_compute_idle();
            if dispatch_profile {
                commands.populate_dispatch_span_boundaries(&mut dispatch_boundaries)?;
            }
            let command_dwords = commands.len_dwords();
            // HIPFIRE_REDLINE_IB_POOL=vmem: allocate the retained indirect
            // buffer from a GPU-agent (VRAM) pool so the command processor
            // fetches the tape from VRAM instead of re-reading it over the
            // host interface on every replay. Host-read surfaces
            // (timestamps, completion signals) stay on the CPU-agent pool.
            // Falls back to the host pool with a warning when no device-local
            // pool exists.
            static IB_VMEM: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                hipfire_config::developer_var("HIPFIRE_REDLINE_IB_POOL").as_deref()
                    == Ok("vmem")
            });
            let ib_pool = if *IB_VMEM {
                match KernargPool::discover_device_local(&device) {
                    Ok(p) => Some(p),
                    Err(error) => {
                        eprintln!(
                            "[redline] HIPFIRE_REDLINE_IB_POOL=vmem but no device-local pool \
                             found ({error}); using host pool"
                        );
                        None
                    }
                }
            } else {
                None
            };
            let graph = commands.create_graph(
                &device,
                &pool,
                ib_pool.as_ref(),
                cu_mask.as_ref(),
                dispatch_profile,
            )?;
            (PreparedPm4Graph::Single(graph), command_dwords)
        } else {
            let min_parallel_width = pm4_min_parallel_width_from_config();
            let min_parallel_workgroups = pm4_min_parallel_workgroups_from_config();
            let max_parallel_phases = pm4_max_parallel_phases_from_config();
            let native_phase_sync = pm4_native_phase_sync_from_config();
            let plans = pm4_phase_plan(
                &self.recorded[..prefix],
                min_parallel_width,
                min_parallel_workgroups,
                max_parallel_phases,
            );
            let parallel_phases = plans.iter().filter(|phase| phase.parallel).count();
            let max_width = plans
                .iter()
                .map(|phase| phase.indices.len())
                .max()
                .unwrap_or(1);
            let parallel_launches = plans
                .iter()
                .filter(|phase| phase.parallel)
                .map(|phase| phase.indices.len())
                .sum::<usize>();
            let mut phase_commands = Vec::<Vec<Pm4Commands>>::with_capacity(plans.len());
            let mut command_dwords = 0_u32;
            let mut max_queue_count = 1_usize;

            for phase in &plans {
                let lane_count = if phase.parallel {
                    self.pm4_queue_policy
                        .resolve(device.name(), phase.indices.len())
                } else {
                    1
                };
                if lane_count > 1 {
                    let mut lanes = (0..lane_count)
                        .map(|_| {
                            let mut commands = Pm4Commands::new(
                                pm4_architecture,
                                self.pm4_register_policy,
                                dispatch_initiator_policy,
                                dispatch_interleave,
                                resource_limits_policy,
                            );
                            commands.acquire_entry(gfx12_gcr_trim, entry_acquire_policy);
                            commands
                        })
                        .collect::<Vec<_>>();
                    for (position, index) in phase.indices.iter().copied().enumerate() {
                        lanes[position % lane_count]
                            .dispatch(
                                &kernels[index],
                                geometries[index],
                                self.recorded[index].shared_mem,
                                kernargs[index].address(),
                            )
                            .map_err(|error| format!("{}: {error}", self.recorded[index].kernel))?;
                    }
                    for commands in &mut lanes {
                        commands.wait_compute_idle();
                        command_dwords = command_dwords
                            .checked_add(commands.len_dwords())
                            .ok_or_else(|| "PM4 command dword count overflow".to_owned())?;
                    }
                    max_queue_count = max_queue_count.max(lanes.len());
                    phase_commands.push(lanes);
                    continue;
                }

                let mut commands = Pm4Commands::new(
                    pm4_architecture,
                    self.pm4_register_policy,
                    dispatch_initiator_policy,
                    dispatch_interleave,
                    resource_limits_policy,
                );
                commands.acquire_entry(gfx12_gcr_trim, entry_acquire_policy);
                let mut resource_frontier = ResourceFrontier::default();
                for (position, index) in phase.indices.iter().copied().enumerate() {
                    if position != 0 && !phase.parallel {
                        let previous_index = phase.indices[position - 1];
                        let previous_launch = &self.recorded[previous_index];
                        let current_launch = &self.recorded[index];
                        let previous = previous_launch.kernel.as_str();
                        let current = current_launch.kernel.as_str();
                        let allowlist_independent = independent_sibling(previous, current);
                        let resources_independent = resource_frontier.independent(current_launch);
                        let independent = match self.pm4_wait_policy {
                            Pm4WaitPolicy::Allowlist | Pm4WaitPolicy::ResourceAudit => {
                                allowlist_independent
                            }
                            Pm4WaitPolicy::Resource => resources_independent,
                        };
                        if !independent {
                            commands.wait_compute_idle();
                        }
                        resource_frontier.advance(current_launch, resources_independent);
                        if (!independent && commands.requires_dependency_acquire())
                            || self
                                .pm4_mid_acquire_policy
                                .acquire_between(previous, current)
                        {
                            commands.acquire_inter_node(
                                gfx12_gcr_trim,
                                pm4_vmem_acquire_enabled(
                                    pm4_architecture,
                                    gfx11_vmem_acquire,
                                    current,
                                ),
                            );
                        }
                    } else {
                        resource_frontier.advance(&self.recorded[index], false);
                    }
                    commands
                        .dispatch(
                            &kernels[index],
                            geometries[index],
                            self.recorded[index].shared_mem,
                            kernargs[index].address(),
                        )
                        .map_err(|error| format!("{}: {error}", self.recorded[index].kernel))?;
                }
                commands.wait_compute_idle();
                command_dwords = command_dwords
                    .checked_add(commands.len_dwords())
                    .ok_or_else(|| "PM4 command dword count overflow".to_owned())?;
                phase_commands.push(vec![commands]);
            }

            let graph = create_phased_pm4_graph(
                pm4_architecture,
                &device,
                &pool,
                &phase_commands,
                native_phase_sync,
            )?;
            debug_assert_eq!(graph.queue_count(), max_queue_count);
            eprintln!(
                "[redline] PM4 phase plan architecture={} queues={} phases={} parallel_phases={} parallel_launches={} max_width={} min_parallel_width={} min_parallel_workgroups={} max_parallel_phases={} sync={}",
                device.name(),
                graph.queue_count(),
                graph.phase_count(),
                parallel_phases,
                parallel_launches,
                max_width,
                min_parallel_width,
                min_parallel_workgroups,
                max_parallel_phases,
                if native_phase_sync { "native" } else { "aql" },
            );
            (PreparedPm4Graph::Phased(graph), command_dwords)
        };
        let queue_id = graph.queue_id();
        self.prepared_pm4 = Some(PreparedPm4Replay {
            graph,
            _kernels: kernels,
            kernargs,
            dynamic_gdn_frames,
            dynamic_grids,
            pm4_architecture,
            dispatch_count: prefix,
            command_dwords,
            dispatch_boundaries: dispatch_profile.then_some(dispatch_boundaries),
        });
        self.state = ReplayState::Ready;
        Ok((prefix, command_dwords, queue_id))
    }

    /// # Safety
    ///
    /// The captured model allocations and all pointed-to buffers must still be
    /// live and in the same binding layout.
    pub unsafe fn replay_linear_aql(&mut self, position: usize) -> Result<GpuBatchTiming, String> {
        let result = {
            let prepared = self
                .prepared
                .as_mut()
                .ok_or_else(|| "no prepared AQL replay".to_owned())?;
            // SAFETY: forwarded from the model owner.
            unsafe { prepared.replay_and_wait() }
        };
        self.observe_replay_result(position, result)
    }

    /// # Safety
    ///
    /// The captured model allocations and all pointed-to buffers must still be
    /// live and in the same binding layout.
    pub unsafe fn replay_pm4(&mut self, position: usize) -> Result<GpuMultiQueueTiming, String> {
        let result = {
            let prepared = self
                .prepared_pm4
                .as_mut()
                .ok_or_else(|| "no prepared PM4 replay".to_owned())?;
            // SAFETY: forwarded from the model owner.
            unsafe { prepared.replay_and_wait(position) }
        };
        self.observe_replay_result(position, result)
    }

    /// Replay one explicitly instrumented PM4 graph exactly once.
    ///
    /// # Safety
    ///
    /// Captured model allocations and pointed-to buffers must remain live.
    pub unsafe fn replay_pm4_dispatch_profile(
        &mut self,
        position: usize,
    ) -> Result<Pm4DispatchProfile, String> {
        let result = {
            let prepared = self
                .prepared_pm4
                .as_mut()
                .ok_or_else(|| "no prepared PM4 replay".to_owned())?;
            // SAFETY: forwarded from the model owner.
            unsafe { prepared.replay_and_wait_dispatch_profiled(position) }
        };
        self.observe_replay_result(position, result)
    }

    pub fn prepared_pm4_dispatch_boundaries(&self) -> Option<&[Pm4DispatchBoundary]> {
        self.prepared_pm4
            .as_ref()
            .and_then(PreparedPm4Replay::dispatch_boundaries)
    }
    fn observe_replay_result<T>(
        &mut self,
        position: usize,
        result: Result<T, String>,
    ) -> Result<T, String> {
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                self.replay_observation.failed = true;
                return Err(error);
            }
        };
        self.replay_observation.count = self.replay_observation.count.saturating_add(1);
        self.replay_observation
            .first_position
            .get_or_insert(position);
        self.replay_observation.last_position = Some(position);
        Ok(value)
    }

    /// Start one explicitly delimited prefill or decode capture. This clears
    /// only the prior launch sequence; validation observations and the backend
    /// request remain intact.
    pub fn begin_capture(&mut self) -> Result<(), &'static str> {
        match self.state {
            ReplayState::Hip => return Err("replay backend is disabled"),
            ReplayState::Fallback => return Err("replay controller is in sticky fallback"),
            ReplayState::Ready => return Err("cannot capture after a prepared plan is installed"),
            _ => {}
        }
        self.recorded.clear();
        self.state = ReplayState::RecordingWarmup;
        Ok(())
    }

    /// Close the current explicit capture and retain its sequence for
    /// fingerprinting/adapter construction. No launch route changes here.
    pub fn finish_capture(&mut self) -> Result<ReplayCaptureSummary, &'static str> {
        if self.state != ReplayState::RecordingWarmup {
            return Err("no replay capture is active");
        }
        let summary = self.capture_summary();
        self.state = if self.certified_speedups.len() >= 2 {
            ReplayState::ShadowValidated
        } else {
            ReplayState::Captured
        };
        Ok(summary)
    }

    pub fn capture_summary(&self) -> ReplayCaptureSummary {
        let unique_kernel_count = self
            .recorded
            .iter()
            .map(|launch| launch.kernel.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let mut hash = 0xcbf29ce484222325_u64;
        for launch in &self.recorded {
            for byte in launch.kernel.as_bytes().iter().copied().chain([0]) {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            for value in launch
                .grid
                .iter()
                .chain(&launch.block)
                .chain([&launch.shared_mem])
            {
                for byte in value.to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
            match launch.grid_binding {
                None => {
                    hash ^= 0;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                Some(ReplayGridBinding::PositionCeilDiv {
                    axis,
                    addend,
                    divisor,
                }) => {
                    hash ^= 1;
                    hash = hash.wrapping_mul(0x100000001b3);
                    for byte in [axis]
                        .into_iter()
                        .chain(addend.to_le_bytes())
                        .chain(divisor.to_le_bytes())
                    {
                        hash ^= u64::from(byte);
                        hash = hash.wrapping_mul(0x100000001b3);
                    }
                }
            }
        }
        ReplayCaptureSummary {
            launch_count: self.recorded.len(),
            unique_kernel_count,
            sequence_hash: hash,
        }
    }

    pub(crate) fn record_hip_launch_typed_bound(
        &mut self,
        hip: &HipRuntime,
        kernel: &str,
        artifact: Option<PathBuf>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
        grid_binding: Option<ReplayGridBinding>,
    ) {
        let accesses = recorded_resource_accesses(hip, kernel, kernarg);
        self.record_hip_launch_with_accesses(
            kernel,
            artifact,
            grid,
            block,
            shared_mem,
            kernarg,
            grid_binding,
            accesses,
        );
    }

    #[cfg(test)]
    fn record_hip_launch(
        &mut self,
        kernel: &str,
        artifact: Option<PathBuf>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
    ) {
        self.record_hip_launch_with_accesses(
            kernel, artifact, grid, block, shared_mem, kernarg, None, None,
        );
    }

    fn record_hip_launch_with_accesses(
        &mut self,
        kernel: &str,
        artifact: Option<PathBuf>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
        grid_binding: Option<ReplayGridBinding>,
        accesses: Option<Vec<RecordedResourceAccess>>,
    ) {
        if !self.is_recording() {
            return;
        }
        if self.recorded.len() == self.max_recorded_launches {
            self.fallback("warmup launch recorder capacity exceeded");
            return;
        }
        self.recorded.push(RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact,
            grid,
            block,
            shared_mem,
            grid_binding,
            kernarg: kernarg.to_vec(),
            accesses,
        });
    }

    pub fn observe_shadow(&mut self, observation: ShadowValidation) {
        if self.state == ReplayState::Hip || self.state == ReplayState::Fallback {
            return;
        }
        if !observation.passes(self.threshold) {
            self.fallback("shadow parity, ABI, timing, or speed threshold failed");
            return;
        }
        self.certified_speedups.push(observation.speedup_over_hip);
        if self.certified_speedups.len() >= 2 {
            self.state = ReplayState::ShadowValidated;
        }
    }

    /// Mark that a model adapter has converted recorded launches into an
    /// explicit hazard-checked `redline_dispatch::CompiledPlan`, prepared it,
    /// and retained HIP buffers/artifacts for its lifetime.
    pub fn install_prepared_plan(&mut self) -> Result<(), &'static str> {
        if self.state != ReplayState::ShadowValidated {
            return Err("two passing shadow validations are required");
        }
        if self.request == ReplayBackendRequest::Shadow {
            return Err("shadow mode never changes the launch route");
        }
        self.state = ReplayState::Ready;
        Ok(())
    }

    pub fn should_route_aql(&self) -> bool {
        self.forward_eligible
            && self.request == ReplayBackendRequest::Auto
            && self.state == ReplayState::Ready
            && self.transport == ReplayTransport::AqlPackets
    }

    pub fn should_route_pm4(&self) -> bool {
        self.forward_eligible
            && self.request == ReplayBackendRequest::Auto
            && self.state == ReplayState::Ready
            && self.transport == ReplayTransport::Pm4Ib
    }

    pub fn uses_pm4_transport(&self) -> bool {
        self.transport == ReplayTransport::Pm4Ib
    }

    pub fn poison(&mut self, reason: impl Into<String>) {
        self.fallback_reason = Some(reason.into());
        self.state = ReplayState::Fallback;
    }

    fn fallback(&mut self, reason: &str) {
        self.poison(reason);
    }
}

fn populate_gfx12_kernarg(
    destination: &mut KernargBuffer,
    launch: &RecordedHipLaunch,
    loader_bytes: usize,
) -> Result<(), String> {
    let (base, has_implicit) = validate_loader_kernarg(launch, loader_bytes)?;
    if destination.len() != loader_bytes {
        return Err(format!(
            "{}: destination {} bytes != loader {loader_bytes}",
            launch.kernel,
            destination.len(),
        ));
    }
    let bytes = destination.as_mut_bytes();
    bytes.fill(0);
    bytes[..base].copy_from_slice(&launch.kernarg[..base]);

    if !has_implicit {
        return Ok(());
    }

    for axis in 0..3 {
        put_u32(bytes, base + axis * 4, launch.grid[axis])?;
        let group = u16::try_from(launch.block[axis]).map_err(|_| {
            format!(
                "{}: workgroup dimension {} exceeds u16",
                launch.kernel, launch.block[axis]
            )
        })?;
        put_u16(bytes, base + 12 + axis * 2, group)?;
        // HIP's grid values are work-group counts, so total work-items are an
        // exact multiple of the group size and every remainder is zero.
        put_u16(bytes, base + 18 + axis * 2, 0)?;
    }
    let dimensions = if launch.grid[2] != 1 || launch.block[2] != 1 {
        3
    } else if launch.grid[1] != 1 || launch.block[1] != 1 {
        2
    } else {
        1
    };
    put_u16(bytes, base + 64, dimensions)?;
    put_u32(bytes, base + 120, launch.shared_mem)?;
    Ok(())
}

fn validate_loader_kernarg(
    launch: &RecordedHipLaunch,
    loader_bytes: usize,
) -> Result<(usize, bool), String> {
    const IMPLICIT_BYTES: usize = 256;
    let captured = launch.kernarg.len();
    let (explicit, has_implicit) = if loader_bytes <= captured {
        (loader_bytes, false)
    } else {
        let explicit = loader_bytes.checked_sub(IMPLICIT_BYTES).ok_or_else(|| {
            format!(
                "loader requires {loader_bytes} bytes, larger than captured {captured} but smaller than implicit suffix"
            )
        })?;
        if explicit > captured {
            return Err(format!(
                "loader explicit prefix {explicit} exceeds captured {captured} bytes"
            ));
        }
        (explicit, true)
    };
    if launch.kernarg[explicit..].iter().any(|byte| *byte != 0) {
        return Err(format!(
            "loader explicit prefix {explicit} would discard nonzero captured bytes from {}",
            launch.kernarg.len(),
        ));
    }
    Ok((explicit, has_implicit))
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) -> Result<(), String> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| "kernarg u16 offset overflow".to_owned())?;
    let slot = bytes
        .get_mut(offset..end)
        .ok_or_else(|| format!("kernarg u16 write {offset}..{end} is out of bounds"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "kernarg u32 offset overflow".to_owned())?;
    let slot = bytes
        .get_mut(offset..end)
        .ok_or_else(|| format!("kernarg u32 write {offset}..{end} is out of bounds"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A3B_REPLAY_KERNELS: &[&str] = &[
        "fused_rmsnorm_mq_rotate",
        "fused_rmsnorm_mq_rotate_vecsum",
        "fused_rmsnorm_mq_rotate_vecsum_sign_const",
        "fused_rmsnorm_mq_rotate_vecsum_sign_lds",
        "fused_rmsnorm_mq_rotate_wavegrid",
        "rmsnorm_reduce_gfx1100",
        "rotate_with_rms_gfx1100",
        "fused_qkvza_hfq4g256",
        "fused_qkvza_hfq4g256_k2048",
        "fused_qkvza_hfq4g256_k2048_r2",
        "fused_qkvza_hfq4g256_k2048_cpol_slc",
        "fused_qkvza_hfq4g256_k2048_scalar_prep",
        "fused_qkvza_hfq4g256_wavepack4",
        "fused_qkvza_hfq4g256_ldsx8",
        "fused_qkvza_hfq4g256_reduce_chain",
        "fused_sigmoid_alpha_gate_f32",
        "conv1d_silu_split_f32",
        "conv1d_silu_split_qknorm_b256_scalar_prep",
        "fused_qk_l2_norm_scale_f32",
        "repeat_interleave_qk_f32",
        "gated_delta_net_q8_fast",
        "gated_norm_f32",
        "gated_norm_mq_rotate_gfx1100",
        "gated_norm_mq_rotate_gfx1151",
        "qwen35_fa_prep_gfx1100",
        "qwen35_fa_prep_gfx1151",
        "mq_rotate_x",
        "gemv_hfq4g256_residual",
        "gemv_hfq4g256_residual_cpol_rt",
        "gemv_hfq4g256_residual_cpol_rt_low",
        "gemv_hfq4g256_residual_cpol_slc",
        "gemv_hfq4g256_residual_k2048",
        "gemv_hfq4g256_residual_rt_low_gfx1151",
        "gemv_hfq4g256",
        "gemv_hfq4g256_k2048",
        "gemv_hfq4g256_wide",
        "softmax_f32",
        "moe_topk_renorm_k8",
        "moe_router_softmax_topk_k8_wave64",
        "moe_router_softmax_topk_k8_wave64_exact",
        "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate",
        "fused_silu_mul_mq_rotate",
        "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
        "gemv_hfq4g256_moe_gate_up_k8_indexed",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
        "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
        "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151",
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc",
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151",
        "gemv_hfq4g256_moe_down_k8_indexed_last_combine",
        "moe_down_combine_k8_batched",
        "moe_down_combine_k8_batched_vec4",
        "moe_down_combine_rmsnorm_mq_rotate_vecsum",
        "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151",
        "fused_qkv_hfq4g256",
        "deinterleave_f32",
        "rmsnorm_f32",
        "rope_partial_halfsplit_f32",
        "kv_cache_write_asym_k_fwht3",
        "kv_cache_write_q8_0",
        "kv_cache_write_q8_0_pair",
        "attention_flash_fwht3_tile",
        "attention_flash_q8_0_reduce",
        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",
        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",
        "sigmoid_mul_f32",
        "gemv_hfq4g256_multirow_r2",
        "gemv_hfq4g256_multirow_r4",
        "gemv_hfq4g256_multirow_r8",
    ];

    fn passing(speedup: f64) -> ShadowValidation {
        ShadowValidation {
            bit_exact: true,
            guards_intact: true,
            same_artifact: true,
            abi_valid: true,
            automatic_clocks: true,
            gpu_timed: true,
            speedup_over_hip: speedup,
        }
    }

    #[test]
    fn radiowave_vmem_cache_classification_fails_closed() {
        assert!(radiowave_vmem_only_consumer("fused_rmsnorm_mq_rotate"));
        assert!(radiowave_vmem_only_consumer(
            "fused_rmsnorm_mq_rotate_vecsum"
        ));
        assert!(radiowave_vmem_only_consumer(
            "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"
        ));
        assert!(radiowave_vmem_only_consumer("mq_rotate_x"));
        assert!(!radiowave_vmem_only_consumer(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu"
        ));
        assert!(radiowave_vmem_only_consumer(
            "gemv_hfq4g256_residual_rt_low_gfx1151"
        ));
        assert!(!radiowave_vmem_only_consumer("fused_qkvza_hfq4g256"));
        assert!(!radiowave_vmem_only_consumer("unknown_kernel"));
    }

    #[test]
    fn gfx12_never_reports_gfx11_vmem_acquire() {
        let kernel = "fused_rmsnorm_mq_rotate";
        assert!(pm4_vmem_acquire_enabled(
            Pm4Architecture::Gfx11,
            true,
            kernel
        ));
        assert!(!pm4_vmem_acquire_enabled(
            Pm4Architecture::Gfx12,
            true,
            kernel
        ));
    }

    #[test]
    fn gfx1151_radiowave_symbols_keep_resource_contracts() {
        let gate = "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151";
        let gate_hybrid = "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151";
        let gate_up_paired = "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151";
        let gate_up_persistent = "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151";
        let qkvza = "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151";
        let qkvza_hybrid = "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151";
        let qkvza_r4 = "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151";
        assert_eq!(expected_kernarg_bytes(gate), Some(48));
        assert_eq!(pointer_effects(gate).map(|effects| effects.len()), Some(5));
        assert_eq!(expected_kernarg_bytes(gate_hybrid), Some(48));
        assert_eq!(
            pointer_effects(gate_hybrid).map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(expected_kernarg_bytes(gate_up_paired), Some(48));
        assert_eq!(
            pointer_effects(gate_up_paired).map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(expected_kernarg_bytes(gate_up_persistent), Some(48));
        assert_eq!(
            pointer_effects(gate_up_persistent).map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(expected_kernarg_bytes(qkvza), Some(96));
        assert_eq!(pointer_effects(qkvza).map(|effects| effects.len()), Some(9));
        assert_eq!(expected_kernarg_bytes(qkvza_hybrid), Some(96));
        assert_eq!(
            pointer_effects(qkvza_hybrid).map(|effects| effects.len()),
            Some(9)
        );
        assert!(!radiowave_vmem_only_consumer(qkvza_hybrid));
        assert_eq!(expected_kernarg_bytes(qkvza_r4), Some(96));
        assert_eq!(
            pointer_effects(qkvza_r4).map(|effects| effects.len()),
            Some(9)
        );
        assert!(!radiowave_vmem_only_consumer(qkvza_r4));
        for producer in [
            "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
        ] {
            assert_eq!(expected_kernarg_bytes(producer), Some(48));
            let effects = pointer_effects(producer).expect("split projection contract");
            assert_eq!(effects.len(), 4);
            assert_eq!(effects[0].mode, RecordedAccessMode::Read);
            assert_eq!(effects[1].mode, RecordedAccessMode::Read);
            assert_eq!(effects[2].mode, RecordedAccessMode::Read);
            assert_eq!(effects[3].mode, RecordedAccessMode::Write);
            assert_eq!(effects[3].offset, 24);
        }
        for (symbol, kernarg_bytes, pointer_count) in [
            ("gated_norm_mq_rotate_gfx1151", 64, 6),
            ("qwen35_fa_prep_gfx1151", 64, 7),
            ("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151", 64, 6),
            ("moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151", 72, 7),
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(kernarg_bytes));
            assert_eq!(
                pointer_effects(symbol).map(|effects| effects.len()),
                Some(pointer_count)
            );
        }
        assert_eq!(
            expected_kernarg_bytes("attention_flash_q8_0_tile"),
            Some(80)
        );
        assert_eq!(
            pointer_effects("attention_flash_q8_0_tile").map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(
            expected_kernarg_bytes("gemv_hfq4g256_residual_wave64"),
            Some(32)
        );
        assert_eq!(
            pointer_effects("gemv_hfq4g256_residual_wave64").map(|effects| effects.len()),
            Some(3)
        );
        let residual_r2 = "gemv_hfq4g256_residual_multirow_r2_gfx1151";
        assert_eq!(expected_kernarg_bytes(residual_r2), Some(32));
        assert_eq!(
            pointer_effects(residual_r2).map(|effects| effects.len()),
            Some(3)
        );
        let residual_k4096 = "gemv_hfq4g256_residual_k4096_gfx1151";
        assert_eq!(expected_kernarg_bytes(residual_k4096), Some(32));
        assert_eq!(
            pointer_effects(residual_k4096).map(|effects| effects.len()),
            Some(3)
        );
        let residual_rt_low = "gemv_hfq4g256_residual_rt_low_gfx1151";
        assert_eq!(expected_kernarg_bytes(residual_rt_low), Some(32));
        assert_eq!(
            pointer_effects(residual_rt_low).map(|effects| effects.len()),
            Some(3)
        );
        assert!(radiowave_vmem_only_consumer(residual_rt_low));
        let down = "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151";
        assert_eq!(expected_kernarg_bytes(down), Some(48));
        assert_eq!(pointer_effects(down).map(|effects| effects.len()), Some(4));
        let down_row8 = "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151";
        assert_eq!(expected_kernarg_bytes(down_row8), Some(48));
        assert_eq!(
            pointer_effects(down_row8).map(|effects| effects.len()),
            Some(4)
        );
        let lm_head = "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151";
        assert_eq!(expected_kernarg_bytes(lm_head), Some(32));
        assert_eq!(
            pointer_effects(lm_head).map(|effects| effects.len()),
            Some(3)
        );
        let lm_head_dot2 = "gemv_hfq4g256_lm_head_dot2_gfx1151";
        assert_eq!(expected_kernarg_bytes(lm_head_dot2), Some(32));
        assert_eq!(
            pointer_effects(lm_head_dot2).map(|effects| effects.len()),
            Some(3)
        );
    }

    #[test]
    fn moe_shared_down_and_routed_gate_up_are_independent_siblings() {
        assert!(independent_sibling(
            "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
        ));
        for kernel in [
            "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
        ] {
            assert!(independent_sibling(
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                kernel,
            ));
        }
        assert!(!independent_sibling(
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
            "fused_silu_mul_mq_rotate",
        ));
    }

    #[test]
    fn pm4_mid_acquire_policies_preserve_required_boundaries() {
        assert_eq!(
            Pm4MidAcquirePolicy::from_value("conservative"),
            Some(Pm4MidAcquirePolicy::Conservative)
        );
        assert_eq!(
            Pm4MidAcquirePolicy::from_value("entry-only"),
            Some(Pm4MidAcquirePolicy::EntryOnly)
        );
        assert_eq!(
            Pm4MidAcquirePolicy::from_value("required-only"),
            Some(Pm4MidAcquirePolicy::RequiredOnly)
        );
        assert!(Pm4MidAcquirePolicy::Conservative
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(!Pm4MidAcquirePolicy::EntryOnly
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(!Pm4MidAcquirePolicy::Conservative.acquire_between("rmsnorm_f32", "gemv_hfq4g256"));
        assert!(!Pm4MidAcquirePolicy::WithoutRope
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(Pm4MidAcquirePolicy::WithoutRope
            .acquire_between("repeat_interleave_qk_f32", "rope_partial_halfsplit_f32"));
        assert!(
            !Pm4MidAcquirePolicy::WithoutMqRotate.acquire_between("mq_rotate_x", "gemv_hfq4g256")
        );
        assert!(Pm4MidAcquirePolicy::RequiredOnly
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(Pm4MidAcquirePolicy::RequiredOnly
            .acquire_between("fused_silu_mul_mq_rotate", "gemv_hfq4g256_residual"));
        assert!(Pm4MidAcquirePolicy::RequiredOnly.acquire_between(
            "fused_silu_mul_mq_rotate",
            "gemv_hfq4g256_residual_k2048_gfx1201"
        ));
        assert!(!Pm4MidAcquirePolicy::RequiredOnly
            .acquire_between("fused_silu_mul_mq_rotate", "gemv_hfq4g256"));
        assert!(Pm4MidAcquirePolicy::RequiredOnly.acquire_between(
            "fused_qk_l2_norm_scale_f32",
            "gated_delta_net_q8_compact2_b2"
        ));
        assert_eq!(Pm4MidAcquirePolicy::from_value("invalid"), None);
    }

    #[test]
    fn gfx1151_dispatch_initiator_policy_is_exact_arch_only() {
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "order",),
            Some(Gfx10DispatchInitiatorPolicy::OrderMode)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "radv",),
            Some(Gfx10DispatchInitiatorPolicy::Radv)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx11, "gfx1100", "radv",),
            Some(Gfx10DispatchInitiatorPolicy::Legacy)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx10, "gfx1030", "radv",),
            Some(Gfx10DispatchInitiatorPolicy::Legacy)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(
                Pm4Architecture::Gfx11,
                "gfx1151",
                "invalid",
            ),
            None
        );
    }

    #[test]
    fn gfx1151_dispatch_interleave_is_exact_arch_only() {
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "64"),
            Some(Some(Gfx11DispatchInterleave::Threads64))
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "0"),
            Some(Some(Gfx11DispatchInterleave::Disabled))
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "inherit",),
            Some(None)
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1100", "64"),
            Some(None)
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx12, "gfx1201", "64"),
            Some(None)
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "32"),
            None
        );
    }

    #[test]
    fn gfx1151_resource_limits_policy_is_exact_arch_only() {
        let radv = Gfx11ComputeResourceLimitsPolicy::Radv {
            force_simd_dist_for_single_wave: false,
        };
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "radv",),
            Some(radv)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(
                Pm4Architecture::Gfx11,
                "gfx1151",
                "simd-always",
            ),
            Some(Gfx11ComputeResourceLimitsPolicy::SimdDestAlways)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx11, "gfx1100", "radv",),
            Some(Gfx11ComputeResourceLimitsPolicy::Legacy)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx12, "gfx1201", "radv",),
            Some(Gfx11ComputeResourceLimitsPolicy::Legacy)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "invalid",),
            None
        );
    }

    #[test]
    fn gfx1151_cu_mask_is_exact_arch_and_wgp_paired() {
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "all"),
            Some(None)
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "32"),
            Some(Some([u32::MAX, 0]))
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "36"),
            Some(Some([u32::MAX, 0xf]))
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1100", "32"),
            Some(None)
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx12, "gfx1201", "32"),
            Some(None)
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "35"),
            None
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "42"),
            None
        );
    }

    #[test]
    fn gfx1151_entry_acquire_is_exact_arch_only() {
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "agent",),
            Some(Gfx11EntryAcquirePolicy::Agent)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "vmem",),
            Some(Gfx11EntryAcquirePolicy::Vmem)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1100", "none",),
            Some(Gfx11EntryAcquirePolicy::System)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx12, "gfx1201", "agent",),
            Some(Gfx11EntryAcquirePolicy::System)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "invalid",),
            None
        );
    }

    #[test]
    fn resource_wait_policy_and_a3b_pointer_catalog_fail_closed() {
        assert_eq!(
            Pm4WaitPolicy::from_value("resource-audit"),
            Some(Pm4WaitPolicy::ResourceAudit)
        );
        assert_eq!(
            expected_kernarg_bytes("gated_delta_net_q8_compact2_b2"),
            Some(96)
        );
        assert!(pointer_effects("gated_delta_net_q8_compact2_b2").is_some());
        assert_eq!(
            expected_kernarg_bytes("conv1d_silu_split_qknorm_b256"),
            Some(80)
        );
        assert!(pointer_effects("conv1d_silu_split_qknorm_b256").is_some());
        assert_eq!(
            expected_kernarg_bytes("moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate"),
            Some(80)
        );
        assert_eq!(
            Pm4WaitPolicy::from_value("resource"),
            Some(Pm4WaitPolicy::Resource)
        );
        assert_eq!(Pm4WaitPolicy::from_value("invalid"), None);
        assert_eq!(
            Pm4RegisterPolicy::from_value("legacy"),
            Some(Pm4RegisterPolicy::Legacy)
        );
        assert_eq!(
            Pm4RegisterPolicy::from_value("1"),
            Some(Pm4RegisterPolicy::Stateful)
        );
        assert_eq!(
            Pm4RegisterPolicy::from_value("static"),
            Some(Pm4RegisterPolicy::Static)
        );
        assert_eq!(Pm4RegisterPolicy::from_value("invalid"), None);
        for kernel in [
            "fused_gate_up_hfq4g256",
            "fused_gate_up_hfq4g256_k1024_gfx1201",
        ] {
            assert_eq!(expected_kernarg_bytes(kernel), Some(64));
            assert_eq!(
                pointer_effects(kernel).map(|effects| effects.len()),
                Some(5)
            );
        }
        assert!(pointer_effects("unknown_kernel").is_none());
        assert!(expected_kernarg_bytes("unknown_kernel").is_none());
        for kernel in A3B_REPLAY_KERNELS {
            let effects = pointer_effects(kernel).unwrap_or_else(|| panic!("missing {kernel}"));
            let kernarg_bytes = expected_kernarg_bytes(kernel)
                .unwrap_or_else(|| panic!("missing ABI size for {kernel}"));
            assert!(!effects.is_empty(), "empty pointer signature for {kernel}");
            assert!(
                effects
                    .iter()
                    .all(|effect| effect.offset + 8 <= kernarg_bytes),
                "pointer offset exceeds kernarg ABI in {kernel}"
            );
            let offsets = effects
                .iter()
                .map(|effect| effect.offset)
                .collect::<BTreeSet<_>>();
            assert_eq!(
                offsets.len(),
                effects.len(),
                "duplicate pointer offset in {kernel}"
            );
        }
    }

    #[test]
    fn allocation_wide_hazards_include_subviews_and_ignore_read_read() {
        let read_a = RecordedResourceAccess {
            allocation_base: 0x1000,
            allocation_bytes: 0x1000,
            access_base: 0x1000,
            mode: RecordedAccessMode::Read,
        };
        let read_same = RecordedResourceAccess {
            allocation_base: 0x1800,
            allocation_bytes: 0x100,
            access_base: 0x1800,
            mode: RecordedAccessMode::Read,
        };
        let write_same = RecordedResourceAccess {
            mode: RecordedAccessMode::Write,
            ..read_same
        };
        let write_other = RecordedResourceAccess {
            allocation_base: 0x3000,
            allocation_bytes: 0x100,
            access_base: 0x3000,
            mode: RecordedAccessMode::Write,
        };
        assert!(!read_a.conflicts(read_same));
        assert!(read_a.conflicts(write_same));
        assert!(!read_a.conflicts(write_other));
    }

    #[test]
    fn exact_start_audit_separates_subviews_from_true_dependencies() {
        let write_left = RecordedResourceAccess {
            allocation_base: 0x1000,
            allocation_bytes: 0x1000,
            access_base: 0x1100,
            mode: RecordedAccessMode::Write,
        };
        let read_right = RecordedResourceAccess {
            access_base: 0x1800,
            mode: RecordedAccessMode::Read,
            ..write_left
        };
        let read_left = RecordedResourceAccess {
            mode: RecordedAccessMode::Read,
            ..write_left
        };

        assert!(write_left.conflicts(read_right));
        assert!(!write_left.same_start_conflicts(read_right));
        assert!(write_left.same_start_conflicts(read_left));
    }

    #[test]
    fn position_grid_binding_narrows_recorded_maximum() {
        let binding = ReplayGridBinding::PositionCeilDiv {
            axis: 1,
            addend: 1,
            divisor: 128,
        };
        assert_eq!(binding.bind(0, [16, 16, 1]).unwrap(), [16, 1, 1]);
        assert_eq!(binding.bind(128, [16, 16, 1]).unwrap(), [16, 2, 1]);
        assert_eq!(binding.bind(2047, [16, 16, 1]).unwrap(), [16, 16, 1]);
        assert_eq!(binding.bind(4095, [16, 16, 1]).unwrap(), [16, 16, 1]);
    }

    #[test]
    fn resource_frontier_catches_non_adjacent_hazards() {
        let launch = |kernel: &str, access: RecordedResourceAccess| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            kernarg: Vec::new(),
            accesses: Some(vec![access]),
        };
        let write_a = launch(
            "write_a",
            RecordedResourceAccess {
                allocation_base: 0x1000,
                allocation_bytes: 0x100,
                access_base: 0x1000,
                mode: RecordedAccessMode::Write,
            },
        );
        let write_b = launch(
            "write_b",
            RecordedResourceAccess {
                allocation_base: 0x2000,
                allocation_bytes: 0x100,
                access_base: 0x2000,
                mode: RecordedAccessMode::Write,
            },
        );
        let read_a = launch(
            "read_a",
            RecordedResourceAccess {
                mode: RecordedAccessMode::Read,
                ..write_a.accesses.as_ref().unwrap()[0]
            },
        );

        let mut frontier = ResourceFrontier::default();
        frontier.advance(&write_a, false);
        assert!(frontier.independent(&write_b));
        frontier.advance(&write_b, true);
        assert!(!frontier.independent(&read_a));
        frontier.advance(&read_a, false);
        assert_eq!(frontier.accesses, read_a.accesses.clone().unwrap());

        let unknown = RecordedHipLaunch {
            accesses: None,
            ..write_b.clone()
        };
        assert!(!frontier.independent(&unknown));
        frontier.advance(&unknown, false);
        assert!(!frontier.independent(&write_b));
    }

    #[test]
    fn pm4_phase_planner_parallelizes_only_pairwise_independent_launches() {
        let launch = |kernel: &str, base: u64, mode: RecordedAccessMode| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            kernarg: Vec::new(),
            accesses: Some(vec![RecordedResourceAccess {
                allocation_base: base,
                allocation_bytes: 0x100,
                access_base: base,
                mode,
            }]),
        };
        let unknown = RecordedHipLaunch {
            accesses: None,
            ..launch("unknown", 0x3000, RecordedAccessMode::Read)
        };
        let recorded = vec![
            launch("write_a", 0x1000, RecordedAccessMode::Write),
            launch("write_b", 0x2000, RecordedAccessMode::Write),
            launch("read_a", 0x1000, RecordedAccessMode::Read),
            launch("write_a_again", 0x1000, RecordedAccessMode::Write),
            unknown,
        ];

        assert_eq!(
            pm4_phase_plan(&recorded, 2, 0, usize::MAX),
            vec![
                Pm4PhasePlan {
                    indices: vec![0, 1],
                    parallel: true,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3, 4],
                    parallel: false,
                },
            ]
        );

        let mut two_parallel_phases = recorded[..2].to_vec();
        two_parallel_phases.push(launch("read_a", 0x1000, RecordedAccessMode::Read));
        two_parallel_phases.push(launch("read_b", 0x2000, RecordedAccessMode::Read));
        assert_eq!(
            pm4_phase_plan(&two_parallel_phases, 2, 0, 1),
            vec![
                Pm4PhasePlan {
                    indices: vec![0, 1],
                    parallel: true,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3],
                    parallel: false,
                },
            ]
        );
    }

    #[test]
    fn pm4_phase_planner_allows_read_read_and_serializes_unknown_accesses() {
        let read = |kernel: &str| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            kernarg: Vec::new(),
            accesses: Some(vec![RecordedResourceAccess {
                allocation_base: 0x1000,
                allocation_bytes: 0x100,
                access_base: 0x1000,
                mode: RecordedAccessMode::Read,
            }]),
        };
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 2, 0, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: true,
            }]
        );
        assert_eq!(
            pm4_phase_plan(
                &[
                    RecordedHipLaunch {
                        accesses: None,
                        ..read("unknown_a")
                    },
                    RecordedHipLaunch {
                        accesses: None,
                        ..read("unknown_b")
                    },
                ],
                2,
                0,
                usize::MAX,
            ),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
            }]
        );
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 3, 0, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
            }]
        );
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 2, 3, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
            }]
        );
    }

    #[test]
    fn default_hip_never_records_or_routes() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Hip);
        controller.record_hip_launch("k", None, [1; 3], [32, 1, 1], 0, &[]);
        assert!(controller.recorded_launches().is_empty());
        assert!(!controller.should_route_aql());
    }

    #[test]
    fn model_default_resets_stale_state_and_is_scoped() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.record_hip_launch("old", None, [1; 3], [32, 1, 1], 0, &[]);
        let mut failed = passing(1.20);
        failed.guards_intact = false;
        controller.observe_shadow(failed);
        assert_eq!(controller.state(), ReplayState::Fallback);

        controller.apply_model_default(true, ReplayTransport::Pm4Ib);
        assert_eq!(controller.request(), ReplayBackendRequest::Auto);
        assert_eq!(controller.state(), ReplayState::Armed);
        assert_eq!(controller.transport_name(), "pm4");
        assert!(controller.recorded_launches().is_empty());
        assert_eq!(controller.fallback_reason(), None);
        controller.begin_auto_capture_if_armed().unwrap();
        assert_eq!(controller.state(), ReplayState::RecordingWarmup);

        controller.apply_model_default(false, ReplayTransport::AqlPackets);
        assert_eq!(controller.request(), ReplayBackendRequest::Hip);
        assert_eq!(controller.state(), ReplayState::Hip);
        assert_eq!(controller.transport_name(), "aql");
        assert!(!controller.is_enabled());
    }

    #[test]
    fn route_observation_records_success_failure_and_resets() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        let failed: Result<(), String> = Err("dispatch failed".to_owned());
        assert!(controller.observe_replay_result(127, failed).is_err());
        assert!(controller.replay_observation().failed);

        controller.begin_replay_observation_window();

        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.observe_replay_result(128, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation {
                count: 2,
                first_position: Some(127),
                last_position: Some(128),
                failed: false,
            }
        );

        controller.apply_model_default(false, ReplayTransport::AqlPackets);
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation::default()
        );
    }

    #[test]
    fn route_observation_windows_are_request_local() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.observe_replay_result(128, Ok(())).unwrap();

        controller.begin_replay_observation_window();
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation::default()
        );

        controller.observe_replay_result(512, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation {
                count: 1,
                first_position: Some(512),
                last_position: Some(512),
                failed: false,
            }
        );
    }

    #[test]
    fn route_proof_marker_is_request_scoped() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.route_proof_log = true;
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );

        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.observe_replay_result(128, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            Some(
                "HIPFIRE_REPLAY_ROUTE_PROOF transport=aql position=127 \
                 request_id=chatcmpl-turn-1 replays=2"
                    .to_owned()
            )
        );
        assert_eq!(controller.replay_observation_marker(""), None);

        controller.begin_replay_observation_window();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-2"),
            None
        );
        controller.observe_replay_result(512, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("invalid request id"),
            None
        );
    }

    #[test]
    fn route_proof_marker_fails_closed_after_replay_error() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.route_proof_log = true;
        controller.observe_replay_result(127, Ok(())).unwrap();
        assert!(controller
            .observe_replay_result::<()>(128, Err("dispatch failed".to_owned()))
            .is_err());

        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
        controller.observe_replay_result(129, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
    }

    #[test]
    fn route_proof_marker_fails_closed_after_request_abort() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.route_proof_log = true;
        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.invalidate_replay_observation_window();

        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
        controller.observe_replay_result(128, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
    }

    #[test]
    fn pm4_packet_identity_fails_closed_for_phased_or_multiqueue() {
        assert_eq!(pm4_packet_identity(1, 1), Some(1));
        assert_eq!(pm4_packet_identity(1, 2), None);
        assert_eq!(pm4_packet_identity(2, 1), None);
        assert_eq!(pm4_packet_identity(2, 2), None);
    }

    #[test]
    fn auto_requires_two_shadows_and_explicit_install() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.record_hip_launch("k", None, [1; 3], [32, 1, 1], 0, &[]);
        controller.observe_shadow(passing(1.08));
        assert_eq!(controller.state(), ReplayState::RecordingWarmup);
        controller.observe_shadow(passing(1.06));
        assert_eq!(controller.state(), ReplayState::ShadowValidated);
        assert!(!controller.should_route_aql());
        controller.install_prepared_plan().unwrap();
        assert!(controller.should_route_aql());
    }

    #[test]
    fn any_failed_gate_is_sticky_fallback() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        let mut failed = passing(1.20);
        failed.guards_intact = false;
        controller.observe_shadow(failed);
        controller.observe_shadow(passing(2.0));
        assert_eq!(controller.state(), ReplayState::Fallback);
        assert!(!controller.should_route_aql());
    }

    #[test]
    fn manual_capture_is_bounded_and_sequence_stable() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Shadow);
        controller.record_hip_launch("ignored", None, [1; 3], [1; 3], 0, &[]);
        assert_eq!(controller.state(), ReplayState::Armed);
        assert!(controller.recorded_launches().is_empty());

        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 2, 3], [32, 1, 1], 0, &[1]);
        controller.record_hip_launch("b", None, [4, 5, 6], [64, 1, 1], 128, &[2]);
        let first = controller.finish_capture().unwrap();
        assert_eq!(controller.state(), ReplayState::Captured);
        assert_eq!(first.launch_count, 2);
        assert_eq!(first.unique_kernel_count, 2);

        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 2, 3], [32, 1, 1], 0, &[1]);
        controller.record_hip_launch("b", None, [4, 5, 6], [64, 1, 1], 128, &[2]);
        assert_eq!(controller.finish_capture().unwrap(), first);

        controller.begin_capture().unwrap();
        controller.record_hip_launch("b", None, [4, 5, 6], [64, 1, 1], 128, &[2]);
        controller.record_hip_launch("a", None, [1, 2, 3], [32, 1, 1], 0, &[1]);
        assert_ne!(
            controller.finish_capture().unwrap().sequence_hash,
            first.sequence_hash
        );
    }

    #[test]
    fn ineligible_forward_neither_records_nor_routes_plain_ar() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.set_forward_eligible(false);
        controller.record_hip_launch("spec", None, [1; 3], [32, 1, 1], 0, &[1]);
        assert!(controller.recorded_launches().is_empty());
        assert!(!controller.should_auto_finalize_capture());

        controller.set_forward_eligible(true);
        controller.record_hip_launch("plain", None, [1; 3], [32, 1, 1], 0, &[2]);
        assert_eq!(controller.recorded_launches().len(), 1);
        controller.observe_shadow(passing(1.08));
        controller.observe_shadow(passing(1.06));
        controller.install_prepared_plan().unwrap();
        assert!(controller.should_route_aql());

        controller.set_forward_eligible(false);
        assert!(!controller.should_route_aql());
    }
}
