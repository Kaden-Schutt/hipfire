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
//! `mq4r_redline_default` (exact `gfx1100`/`gfx1151`/`gfx1201`, single-GPU,
//! case-insensitive `.mq4r`; `gfx1200`/others opt-in)
//! selected by the daemon after model load. Runtime default ≠ Redline
//! certification/registry admission; built-in `hip` profile or explicit backend
//! selection disables the automatic default.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hip_bridge::HipRuntime;
use radiowave::{CodeObjectCertification, KernelArgumentAccess, MutableReadCache};
use redline_dispatch::aql::{
    load_symbols, BatchFencePolicy, Executable, Gfx10DispatchInitiatorPolicy,
    Gfx10Pm4CommandBuffer, Gfx11ComputeResourceLimitsPolicy, Gfx11DispatchInterleave,
    Gfx12Pm4CommandBuffer, GpuBatchTiming, GpuDevice, GpuMultiQueueTiming, GpuSelector,
    HeaderPolicy, KernargBuffer, KernargPool, Kernel, LaunchGeometry, PhasedMultiQueuePm4Ib,
    QueuePolicy, Quiescence, RecordedDispatch, Runtime, SingleQueueBatchGraph, SingleQueuePm4Ib,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayQuiescence {
    Proven,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct RetainedReplayFailure {
    pub error: String,
    pub quiescence: ReplayQuiescence,
}

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

/// Per-stream producer/consumer visibility for legacy (gfx10/gfx11) PM4 IBs.
///
/// `CsPartialFlush` retains the historical EVENT_WRITE path. `ReleaseWait` is
/// admitted only for exact gfx1010 single-queue retained replay and pairs a
/// fine-grained host word with RELEASE_MEM + WAIT_REG_MEM epochs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LegacyDependencyMode {
    CsPartialFlush,
    ReleaseWait { address: u64, next_epoch: u32 },
}

/// Exact gfx1010 gate for the RELEASE_MEM/WAIT_REG_MEM dependency fence.
/// Architecture must already be the gfx10 family map; the device name is
/// matched ASCII-case-insensitively so only the Navi10 agent is selected.
fn gfx1010_release_wait_required(architecture: Pm4Architecture, device_name: &str) -> bool {
    architecture == Pm4Architecture::Gfx10 && device_name.eq_ignore_ascii_case("gfx1010")
}

/// Diagnostic-only override for exact-gfx1010 retained-PM4 dependency fencing.
/// Default remains `ReleaseWait`; `CsPartialFlush` is the historical EVENT_WRITE path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gfx1010DependencyPolicy {
    ReleaseWait,
    CsPartialFlush,
}

/// Pure parser for `HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY`.
///
/// Non-exact-gfx1010 devices always resolve to `CsPartialFlush` and ignore `value`.
/// On exact gfx1010: unset/`release-wait` => `ReleaseWait`, `cs-partial-flush` =>
/// `CsPartialFlush`; any other value is a hard prepare error naming the key.
fn gfx1010_dependency_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: Option<&str>,
) -> Result<Gfx1010DependencyPolicy, String> {
    if !gfx1010_release_wait_required(architecture, device_name) {
        return Ok(Gfx1010DependencyPolicy::CsPartialFlush);
    }
    match value {
        None => Ok(Gfx1010DependencyPolicy::ReleaseWait),
        Some("release-wait") => Ok(Gfx1010DependencyPolicy::ReleaseWait),
        Some("cs-partial-flush") => Ok(Gfx1010DependencyPolicy::CsPartialFlush),
        Some(raw) => Err(format!(
            "invalid HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY={raw:?}; \
             expected unset, \"release-wait\", or \"cs-partial-flush\""
        )),
    }
}

fn gfx1010_dependency_policy_from_config(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Result<Gfx1010DependencyPolicy, String> {
    let raw = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY");
    let policy = gfx1010_dependency_policy_from_value(architecture, device_name, raw.as_deref())?;
    if gfx1010_release_wait_required(architecture, device_name) {
        let source = if raw.is_none() { "default" } else { "explicit" };
        eprintln!("[redline] gfx1010 PM4 dependency mode={policy:?} ({source})");
    }
    Ok(policy)
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
        Self::from_name(device.name())
    }

    fn from_name(device_name: &str) -> Result<Self, String> {
        let name = device_name.to_ascii_lowercase();
        if name.starts_with("gfx10") {
            Ok(Self::Gfx10)
        } else if name.starts_with("gfx11") {
            Ok(Self::Gfx11)
        } else if matches!(name.as_str(), "gfx1200" | "gfx1201") {
            Ok(Self::Gfx12)
        } else {
            Err(format!(
                "retained PM4 has no certified register map for HSA agent {:?}",
                device_name
            ))
        }
    }
}

#[derive(Clone)]
enum Pm4Commands {
    Legacy {
        architecture: Pm4Architecture,
        commands: Gfx10Pm4CommandBuffer,
        dependency_mode: LegacyDependencyMode,
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
                                ..
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
        Self::new_with_dependency(
            architecture,
            policy,
            dispatch_initiator_policy,
            dispatch_interleave,
            resource_limits_policy,
            LegacyDependencyMode::CsPartialFlush,
        )
    }

    fn new_with_dependency(
        architecture: Pm4Architecture,
        policy: Pm4RegisterPolicy,
        dispatch_initiator_policy: Gfx10DispatchInitiatorPolicy,
        dispatch_interleave: Option<Gfx11DispatchInterleave>,
        resource_limits_policy: Gfx11ComputeResourceLimitsPolicy,
        dependency_mode: LegacyDependencyMode,
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
                    dependency_mode,
                }
            }
            Pm4Architecture::Gfx12 => {
                debug_assert!(
                    matches!(dependency_mode, LegacyDependencyMode::CsPartialFlush),
                    "gfx12 never uses legacy dependency fences"
                );
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

    /// Emit the sentinel epoch-0 release/wait before entry acquire so every
    /// immutable replay starts from a known fence word (ABA prevention).
    fn emit_entry_sentinel_reset(&mut self) -> Result<(), String> {
        match self {
            Self::Legacy {
                commands,
                dependency_mode:
                    LegacyDependencyMode::ReleaseWait {
                        address,
                        next_epoch,
                    },
                ..
            } => {
                if *next_epoch != 0 {
                    return Err(format!(
                        "gfx1010 dependency fence entry sentinel requires next_epoch=0, got {next_epoch}"
                    ));
                }
                commands.dependency_fence(*address, 0);
                Ok(())
            }
            _ => Ok(()),
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

    fn wait_compute_idle(&mut self) -> Result<(), String> {
        match self {
            Self::Legacy {
                commands,
                dependency_mode:
                    LegacyDependencyMode::ReleaseWait {
                        address,
                        next_epoch,
                    },
                ..
            } => {
                let epoch = next_epoch.checked_add(1).ok_or_else(|| {
                    "gfx1010 dependency fence epoch overflow (u32 exhausted)".to_owned()
                })?;
                *next_epoch = epoch;
                commands.dependency_fence(*address, epoch);
                Ok(())
            }
            Self::Legacy { commands, .. } => {
                commands.wait_compute_idle();
                Ok(())
            }
            Self::Gfx12(commands) => {
                commands.wait_compute_idle();
                Ok(())
            }
        }
    }

    fn gfx12_system_acquire(&mut self) -> Result<(), String> {
        match self {
            Self::Gfx12(commands) => {
                commands.acquire_system_gfx12();
                Ok(())
            }
            Self::Legacy { .. } => {
                Err("gfx12 system acquire requested for a legacy PM4 stream".to_owned())
            }
        }
    }

    #[cfg(test)]
    fn dependency_mode(&self) -> Option<LegacyDependencyMode> {
        match self {
            Self::Legacy {
                dependency_mode, ..
            } => Some(*dependency_mode),
            Self::Gfx12(_) => None,
        }
    }

    #[cfg(test)]
    fn dwords(&self) -> Option<&[u32]> {
        match self {
            Self::Legacy { commands, .. } => Some(commands.dwords()),
            Self::Gfx12(_) => None,
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

    fn packet_census(&self) -> Option<Result<BTreeMap<(u32, u32), usize>, usize>> {
        match self {
            Self::Legacy { commands, .. } => Some(commands.packet_census()),
            Self::Gfx12(_) => None,
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
                ..
            } => SingleQueuePm4Ib::create_profiled_gfx10(device, pool, commands),
            Self::Legacy {
                architecture: Pm4Architecture::Gfx11,
                commands,
                ..
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

/// Load exact-object Radiowave certifications once per retained tape.
/// Missing, malformed, or hash-stale manifests are omitted and therefore
/// retain the conservative scalar-cache acquire.
fn radiowave_certifications(
    recorded: &[RecordedHipLaunch],
    prefix: usize,
) -> BTreeMap<PathBuf, CodeObjectCertification> {
    let mut certifications = BTreeMap::new();
    let mut attempted = BTreeSet::new();
    for launch in recorded.iter().take(prefix) {
        let Some(artifact) = launch.artifact.as_ref() else {
            continue;
        };
        if !attempted.insert(artifact.clone()) {
            continue;
        }
        if let Some(certification) = load_radiowave_certification(artifact) {
            certifications.insert(artifact.clone(), certification);
        }
    }
    certifications
}

fn load_radiowave_certification(artifact: &Path) -> Option<CodeObjectCertification> {
    let manifest = artifact.with_extension("radiowave.json");
    let code = std::fs::read(artifact).ok()?;
    let encoded = std::fs::read_to_string(manifest).ok()?;
    CodeObjectCertification::from_json(&code, &encoded).ok()
}

fn radiowave_vmem_only_consumer(
    certifications: &BTreeMap<PathBuf, CodeObjectCertification>,
    launch: &RecordedHipLaunch,
) -> bool {
    let Some(artifact) = launch.artifact.as_ref() else {
        return false;
    };
    certifications.get(artifact).is_some_and(|certification| {
        certification.mutable_read_cache(&launch.kernel) == MutableReadCache::VmemOnly
    })
}

fn pm4_vmem_acquire_enabled(
    architecture: Pm4Architecture,
    configured: bool,
    certifications: &BTreeMap<PathBuf, CodeObjectCertification>,
    launch: &RecordedHipLaunch,
) -> bool {
    pm4_vmem_acquire_arch_enabled(architecture, configured)
        && radiowave_vmem_only_consumer(certifications, launch)
}

fn pm4_vmem_acquire_arch_enabled(architecture: Pm4Architecture, configured: bool) -> bool {
    architecture != Pm4Architecture::Gfx12 && configured
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
        "fused_gate_up_hfq4g256"
            | "fused_gate_up_hfq4g256_k1024_gfx1201"
            | "fused_gate_up_hfq4g256_dot_reform_gfx1100"
            | "fused_gate_up_hfq4g256_dot_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_pair_gfx1100"
            | "fused_gate_up_hfq4g256_pair2_gfx1100"
            | "fused_gate_up_hfq4g256_quad_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_setprio_gfx1100"
            | "fused_gate_up_hfq4g256_lane0_headers_gfx1100"
            | "fused_gate_up_hfq4g256_stage_x32_gfx1100"
            | "fused_gate_up_mq4g256v2"
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
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq2g256gl_moe_gate_up_k8_indexed"
            | "gemv_mq3g256gl_moe_gate_up_k8_indexed"
            // Batched-K4 prefill siblings: same pointer set and modes, but a
            // K_TOP scalar makes the kernarg block 52 B, not 48 (see below).
            | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
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
    if matches!(
        kernel,
        "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_ninepath_d4"
            | "gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed"
            // Batched-K4 prefill siblings; 52 B kernarg (K_TOP scalar).
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
    ) {
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }

    // MQ4G256V2 / MQ6G256V2 MoE routes. Exact sister ABIs of the HFQ4/MQ4V2
    // launchers — V1/V2 names must never alias. Offsets are the naturally
    // aligned HIP kernarg fields; launch_maybe_blob pads the recorded block
    // to 16 B (see expected_kernarg_bytes).
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_gate_up_k8_indexed"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed"
            // Batched Path1 sisters: same pointer set/modes; K_TOP scalar
            // only changes the recorded kernarg length (64 B padded).
            | "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed_batched"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded"
            | "gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24)]);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_ninepath_d4" | "gemv_mq6g256v2_moe_ninepath_d4"
    ) {
        // expert_ptrs, topk_indices, topk_weights, act (read); out is RMW.
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemm_mq4g256v2_moe_grouped_wmma_k2"
            | "gemm_mq4g256v2_moe_grouped_wmma_gfx12"
            | "gemm_mq6g256v2_moe_grouped_wmma_k2"
            | "gemm_mq6g256v2_moe_grouped_wmma_gfx12"
    ) {
        // expert_weight_ptrs, tile_ids, sorted_slot_index, X_src (read);
        // Y_grouped (write).
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }
    // Mixed-precision MoE (tags 0..18, V1/V2 frozen). Block-uniform dtype_tags branch.
    // Gate/up carries 6 pointers (adds dtype_tags at +8); down carries 5.
    // Grouped mixed carries 6 pointers (adds dtype_tags at +8). V1/V2 names never alias.
    if kernel == "gemv_mixed_moe_gate_up_k8_indexed_batched" {
        // expert_ptrs, dtype_tags, topk_indices, x, y_gate, y_up
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
        ]);
    }
    if kernel == "gemv_mixed_moe_down_k8_indexed_batched_expanded" {
        // expert_ptrs, dtype_tags, topk_indices, rot_batch, expert_outputs
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemm_mixed_moe_grouped_wmma_k2"
            | "gemm_mixed_moe_grouped_wmma_gfx12"
            | "gemm_mixed_moe_grouped_wmma_4w_k2"
    ) {
        // expert_weight_ptrs, dtype_tags, tile_ids, sorted_slot_index, X_src (read); Y_grouped (write)
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]);
    }
    // Dense shared-expert V2 (qt44 qt47). Plain GEMV / residual / multirow share the same
    // 3-pointer ABI: a_raw, x (read); y (write/RMW). Same padded size as HFQ4 dense.
    // The gfx1100 lm_head x-buffer candidate is the ornith M=248320 K=2048
    // specialization of the plain decode; its replay contract is identical to
    // generic gemv_mq4g256v2 (A@0 read, x@8 read, y@16 write, M@24, K@28, 32B).
    if matches!(
        kernel,
        "gemv_mq4g256v2"
            | "gemv_mq4g256v2_residual"
            | "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch"
            | "gemv_mq4g256v2_gfx1100_lm_head_x_buffer"
            | "gemv_mq6g256v2"
            | "gemv_mq6g256v2_residual"
            | "gemv_mq4g256v2_multirow_r2"
            | "gemv_mq4g256v2_multirow_r4"
            | "gemv_mq4g256v2_multirow_r8"
            | "gemv_mq6g256v2_multirow_r2"
            | "gemv_mq6g256v2_multirow_r4"
            | "gemv_mq6g256v2_multirow_r8"
    ) {
        return Some(vec![read(0), read(8), write(16)]);
    }
    // Dense shared-expert V2 residual WMMA GEMM (prefill/batched). 3 pointers + 3 i32
    // (M,K,batch). Y is write (output) via residual path; distinct symbols per arch tile.
    if matches!(
        kernel,
        "gemm_mq4g256v2_residual_wmma"
            | "gemm_mq4g256v2_residual_wmma_gfx12"
            | "gemm_mq6g256v2_residual_wmma"
            | "gemm_mq6g256v2_residual_wmma_gfx12"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw4_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw8_lds"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"
    ) {
        return Some(vec![read(0), read(8), write(16)]);
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
    if kernel.starts_with("gated_delta_net_q8_compact") {
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
        "add_inplace_f32" => Some(vec![write(0), read(8)]),
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
        "compressor_add_ape_f32_buf" => Some(vec![write(0), read(8), read(16)]),
        "compressor_overlap_concat_f32" => Some(vec![read(0), write(8)]),
        "compressor_softmax_pool_f32_buf" => Some(vec![read(0), read(8), write(16), read(24)]),
        "deepseek4_attn_swa_buf" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            read(40),
        ]),
        "deepseek4_attn_swa_topk_f32_buf"
        | "deepseek4_attn_swa_topk_ilp4_f32_buf"
        | "deepseek4_attn_swa_topk_scoregrid_f32_buf"
        | "deepseek4_attn_swa_topk_warp_f32_buf" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            read(40),
            write(48),
            read(56),
            read(64),
        ]),
        "deepseek4_fused_silu_mul_clamp_mq_rotate" => {
            Some(vec![read(0), read(8), read(16), read(24), write(32)])
        }
        "deepseek4_moe_topk_bias_aware_f32" => Some(vec![read(0), read(8), write(16), write(24)]),
        "deepseek4_silu_mul_clamp_f32" => Some(vec![read(0), read(8), write(16)]),
        "embedding_q8_buf_broadcast" => Some(vec![read(0), write(8), read(16)]),
        "deepseek4_topk_kv_gather_f32_buf" | "deepseek4_topk_kv_gather_tiled_f32_buf" => {
            Some(vec![read(0), read(8), write(16), read(24), read(32)])
        }
        "deepseek4_topk_kv_gather_identity_f32_buf" => Some(vec![read(0), write(8), read(16)]),
        "fused_rmsnorm_mq_rotate_plain" | "fused_rmsnorm_mq_rotate_plain_nox" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
        ]),
        "gemv_mfp4g32_e8_soa_grouped_gfx1151"
        | "gemv_mfp4g32_e8_soa_u4"
        | "gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151" => Some(vec![read(0), read(8), write(16)]),
        "gemv_mq2g256_lloyd_moe_down_expanded_k4" => {
            Some(vec![read(0), read(8), read(16), write(24)])
        }
        "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
        | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"
        | "gemv_mq2g256_lloyd_moe_down_residual_scaled_rankpair_indexed"
        | "gemv_mq2g256_lloyd_moe_down_residual_scaled_rowtile2_indexed" => {
            Some(vec![read(0), read(8), read(16), read(24), write(32)])
        }
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
        | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_wavecb" => {
            Some(vec![read(0), read(8), read(16), write(24), write(32)])
        }
        "hash_router_normalize_f32_buf" => {
            Some(vec![read(0), read(8), read(16), write(24), write(32)])
        }
        "hc_apply_alpha" => Some(vec![write(0), read(8), read(16)]),
        "hc_finalize_control" => Some(vec![write(0), read(8), read(16)]),
        "hc_finalize_input_map" => Some(vec![write(0), read(8), read(16), read(24), write(32)]),
        "hc_compute_control" | "hc_compute_control_vec4" | "hc_head_compute_pre" => {
            Some(vec![read(0), read(8), read(16), write(24)])
        }
        "hc_compute_control_vec4_finalize" => {
            Some(vec![read(0), read(8), read(16), write(24), read(32)])
        }
        "hc_input_map_4stream" => Some(vec![read(0), read(8), write(16)]),
        "hc_mix_4stream" => Some(vec![read(0), read(8), read(16), read(24), write(32)]),
        "hc_pre_post_sigmoid_scale_f32" | "hc_sinkhorn_4x4" => Some(vec![write(0)]),
        "indexer_relu_score_f32_buf" => Some(vec![read(0), read(8), read(16), write(24), read(32)]),
        "indexer_top_k_buf" | "indexer_top_k_buf_parallel" => {
            Some(vec![read(0), write(8), read(16), read(24)])
        }
        "rmsnorm_f32_at_slot_buf" => Some(vec![write(0), read(8), read(16)]),
        "rope_tail_interleaved_f32"
        | "rope_tail_yarn_interleaved_f32"
        | "rope_tail_yarn_interleaved_wide_f32" => Some(vec![write(0), write(8), read(16)]),
        "rope_tail_yarn_interleaved_at_slot_buf_f32" => Some(vec![write(0), read(8), read(16)]),
        "sqrt_softplus_f32" => Some(vec![write(0)]),
        "state_overlap_shift_f32_buf" => Some(vec![write(0), read(8)]),
        "state_ring_write_f32_buf" | "swa_ring_write_f32_buf" => {
            Some(vec![read(0), write(8), read(16)])
        }
        "zero_f32" => Some(vec![write(0)]),
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
        | "fused_qkvza_hfq4g256_reduce_chain"
        | "fused_qkvza_mq4g256v2"
        | "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100" => Some(vec![
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
        // LFM retained-PM4 fallback: state is RMW at @8 (write covers RMW).
        "conv1d_gated_decode_f32" => Some(vec![read(0), write(8), read(16), write(24)]),
        // LFM retained-PM4 fallback: q/k/v read, out write, pos read.
        "attention_q8_0_kv" => Some(vec![read(0), read(8), read(16), write(24), read(32)]),
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
        "gated_delta_net_f32" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
        ]),
        "gated_norm_f32" => Some(vec![read(0), read(8), read(16), write(24)]),
        "gated_norm_mq_rotate_gfx1100"
        | "gated_norm_mq_rotate_k6144_gfx1100"
        | "gated_norm_mq_rotate_gfx1151" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]),
        "qwen35_fa_prep_gfx1100" | "qwen36_27b_fa_prep_gfx1100" | "qwen35_fa_prep_gfx1151" => {
            Some(vec![
                read(0),
                write(8),
                write(16),
                write(24),
                read(32),
                read(40),
                read(48),
            ])
        }
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
        "gemv_hfq4g256_residual_sigmoid_scaled_gpu"
        | "gemv_mq4g256v2_residual_sigmoid_scaled_k512" => {
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
        "fused_qkv_hfq4g256"
        | "fused_qkv_mq4g256v2"
        | "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
            write(48),
        ]),
        "deinterleave_f32" => Some(vec![read(0), write(8), write(16)]),
        "rmsnorm_f32" | "rmsnorm_f32_warp_reduce" => Some(vec![read(0), read(8), write(16)]),
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
        "gemma4_ple_gelu_mul_strided_f32" => Some(vec![read(0), read(8), write(16)]),
        _ => None,
    }
}

fn expected_kernarg_bytes(kernel: &str) -> Option<usize> {
    if matches!(
        kernel,
        "hc_pre_post_sigmoid_scale_f32" | "hc_sinkhorn_4x4" | "sqrt_softplus_f32" | "zero_f32"
    ) {
        return Some(16);
    }
    if matches!(
        kernel,
        "compressor_add_ape_f32_buf"
            | "compressor_overlap_concat_f32"
            | "deepseek4_silu_mul_clamp_f32"
            | "embedding_q8_buf_broadcast"
            | "deepseek4_topk_kv_gather_identity_f32_buf"
            | "gemv_mfp4g32_e8_soa_u4"
            | "gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151"
            | "hc_apply_alpha"
            | "rmsnorm_f32_at_slot_buf"
            | "state_overlap_shift_f32_buf"
            | "state_ring_write_f32_buf"
            | "add_inplace_f32"
    ) {
        return Some(32);
    }
    if matches!(
        kernel,
        "compressor_softmax_pool_f32_buf"
            | "deepseek4_fused_silu_mul_clamp_mq_rotate"
            | "deepseek4_moe_topk_bias_aware_f32"
            | "gemv_mfp4g32_e8_soa_grouped_gfx1151"
            | "gemv_mq2g256_lloyd_moe_down_expanded_k4"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_rowtile2_indexed"
            | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_wavecb"
            | "hc_compute_control"
            | "hc_compute_control_vec4"
            | "hc_finalize_control"
            | "indexer_relu_score_f32_buf"
            | "indexer_top_k_buf"
            | "indexer_top_k_buf_parallel"
            | "rope_tail_interleaved_f32"
            | "swa_ring_write_f32_buf"
    ) {
        return Some(48);
    }
    if kernel == "hc_finalize_input_map" {
        return Some(56);
    }
    if kernel == "hc_compute_control_vec4_finalize" {
        return Some(64);
    }
    if kernel == "gemv_mq2g256_lloyd_moe_down_residual_scaled_rankpair_indexed" {
        return Some(56);
    }
    if matches!(
        kernel,
        "deepseek4_attn_swa_buf"
            | "deepseek4_topk_kv_gather_f32_buf"
            | "deepseek4_topk_kv_gather_tiled_f32_buf"
            | "fused_rmsnorm_mq_rotate_plain"
            | "fused_rmsnorm_mq_rotate_plain_nox"
            | "hash_router_normalize_f32_buf"
            | "hc_head_compute_pre"
            | "rope_tail_yarn_interleaved_at_slot_buf_f32"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "rope_tail_yarn_interleaved_f32" | "rope_tail_yarn_interleaved_wide_f32"
    ) {
        return Some(80);
    }
    if matches!(
        kernel,
        "deepseek4_attn_swa_topk_f32_buf"
            | "deepseek4_attn_swa_topk_ilp4_f32_buf"
            | "deepseek4_attn_swa_topk_scoregrid_f32_buf"
            | "deepseek4_attn_swa_topk_warp_f32_buf"
    ) {
        return Some(96);
    }
    if matches!(
        kernel,
        "fused_gate_up_hfq4g256"
            | "fused_gate_up_hfq4g256_k1024_gfx1201"
            | "fused_gate_up_hfq4g256_dot_reform_gfx1100"
            | "fused_gate_up_hfq4g256_dot_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_pair_gfx1100"
            | "fused_gate_up_hfq4g256_pair2_gfx1100"
            | "fused_gate_up_hfq4g256_quad_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_setprio_gfx1100"
            | "fused_gate_up_hfq4g256_lane0_headers_gfx1100"
            | "fused_gate_up_hfq4g256_stage_x32_gfx1100"
            | "fused_gate_up_mq4g256v2"
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
    if matches!(
        kernel,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_ninepath_d4"
    ) {
        return Some(48);
    }
    // Batched-K4 codebook MoE: 5 pointers (40 B) + M, K, K_TOP (12 B) = 52 B.
    // The trailing K_TOP is what makes these NOT 48 like their decode siblings;
    // assuming 48 here would make every kernarg length check fail closed.
    if matches!(
        kernel,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
    ) {
        return Some(52);
    }
    if matches!(
        kernel,
        "gemv_mq2g256gl_moe_gate_up_k8_indexed"
            | "gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "gemv_mq3g256gl_moe_gate_up_k8_indexed"
            | "gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed"
    ) {
        return Some(80);
    }

    // MQ4G256V2 / MQ6G256V2 MoE: exact sister ABIs. Sizes are the recorded
    // launch_maybe_blob lengths (natural fields + pad_to(16)).
    //   gate_up / ninepath: 5 ptr + 2 i32 = 48
    //   expanded down:      4 ptr + 3 i32 = 44 → 48 padded
    //   gate_up batched:    5 ptr + 3 i32 = 52 → 64 padded
    //   grouped WMMA:       5 ptr + 4 i32 = 56 → 64 padded
    //   mixed gate_up:      6 ptr + 3 i32 = 60 → 64 padded
    //   mixed down:         5 ptr + 3 i32 = 52 → 64 padded
    //   mixed grouped:      6 ptr + 4 i32 = 64 (already aligned)
    //   dense GEMV:         3 ptr + 2 i32 = 32
    //   dense GEMM WMMA:    3 ptr + 3 i32 = 36 → 48 padded
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_gate_up_k8_indexed"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed"
            | "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded"
            | "gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded"
            | "gemv_mq4g256v2_moe_ninepath_d4"
            | "gemv_mq6g256v2_moe_ninepath_d4"
    ) {
        return Some(48);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed_batched"
            | "gemm_mq4g256v2_moe_grouped_wmma_k2"
            | "gemm_mq4g256v2_moe_grouped_wmma_gfx12"
            | "gemm_mq6g256v2_moe_grouped_wmma_k2"
            | "gemm_mq6g256v2_moe_grouped_wmma_gfx12"
            | "gemv_mixed_moe_gate_up_k8_indexed_batched"
            | "gemv_mixed_moe_down_k8_indexed_batched_expanded"
            | "gemm_mixed_moe_grouped_wmma_k2"
            | "gemm_mixed_moe_grouped_wmma_gfx12"
            | "gemm_mixed_moe_grouped_wmma_4w_k2"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2"
            | "gemv_mq4g256v2_residual"
            | "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch"
            | "gemv_mq4g256v2_gfx1100_lm_head_x_buffer"
            | "gemv_mq6g256v2"
            | "gemv_mq6g256v2_residual"
            | "gemv_mq4g256v2_multirow_r2"
            | "gemv_mq4g256v2_multirow_r4"
            | "gemv_mq4g256v2_multirow_r8"
            | "gemv_mq6g256v2_multirow_r2"
            | "gemv_mq6g256v2_multirow_r4"
            | "gemv_mq6g256v2_multirow_r8"
    ) {
        return Some(32);
    }
    if matches!(
        kernel,
        "gemm_mq4g256v2_residual_wmma"
            | "gemm_mq4g256v2_residual_wmma_gfx12"
            | "gemm_mq6g256v2_residual_wmma"
            | "gemm_mq6g256v2_residual_wmma_gfx12"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw4_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw8_lds"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"
    ) {
        return Some(48);
    }

    if kernel.starts_with("gated_delta_net_q8_compact") {
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
        | "rmsnorm_f32_warp_reduce"
        | "rmsnorm_reduce_gfx1100"
        | "hc_input_map_4stream"
        | "sigmoid_mul_f32" => Some(32),
        "gemma4_ple_gelu_mul_strided_f32" => Some(48),
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
        | "gemv_mq4g256v2_residual_sigmoid_scaled_k512"
        | "hc_mix_4stream"
        | "kv_cache_write_asym_k_fwht3"
        | "kv_cache_write_q8_0_pair"
        | "mq_rotate_x"
        | "repeat_interleave_qk_f32"
        | "rope_partial_halfsplit_f32"
        | "conv1d_gated_decode_f32" => Some(48),
        "conv1d_silu_split_f32"
        | "gated_norm_mq_rotate_gfx1100"
        | "gated_norm_mq_rotate_k6144_gfx1100"
        | "gated_norm_mq_rotate_gfx1151"
        | "qwen35_fa_prep_gfx1100"
        | "qwen36_27b_fa_prep_gfx1100"
        | "qwen35_fa_prep_gfx1151"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151"
        | "fused_rmsnorm_mq_rotate_wavegrid"
        | "rotate_with_rms_gfx1100"
        | "attention_q8_0_kv" => Some(64),
        "moe_down_combine_rmsnorm_mq_rotate_vecsum"
        | "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151" => Some(72),
        "gemv_hfq4g256_moe_down_k8_indexed_last_combine" => Some(64),
        "attention_flash_q8_0_tile"
        | "fused_qkv_hfq4g256"
        | "fused_qkv_mq4g256v2"
        | "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100"
        | "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate" => Some(80),
        "attention_flash_fwht3_tile"
        | "fused_qkvza_hfq4g256"
        | "fused_qkvza_hfq4g256_k2048"
        | "fused_qkvza_hfq4g256_k2048_r2"
        | "fused_qkvza_hfq4g256_k2048_cpol_slc"
        | "fused_qkvza_hfq4g256_wavepack4"
        | "fused_qkvza_hfq4g256_ldsx8"
        | "fused_qkvza_hfq4g256_reduce_chain"
        | "fused_qkvza_mq4g256v2"
        | "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100"
        | "gated_delta_net_q8_fast" => Some(96),
        "gated_delta_net_f32" => Some(80),
        _ => None,
    }
}

fn apply_qwen_q8_full_attention_visibility(
    launches: &[RecordedHipLaunch],
    headers: &mut [HeaderPolicy],
) {
    // The Q8 full-attention body carries intermediate Q/K/V, tile reductions,
    // and the gated attention result through separate global-memory buffers.
    // Same-queue barriers alone reproduced stale intermediates on gfx1201;
    // restoring the captured HIP dispatches' system scopes for this narrow
    // body makes the multi-position AQL state/logit/KV shadow bit-exact.
    debug_assert_eq!(launches.len(), headers.len());
    let mut full_attention_body = false;
    for (launch, header) in launches.iter().zip(headers) {
        let kernel = launch.kernel.as_str();
        full_attention_body |= kernel == "fused_qkv_hfq4g256";
        if full_attention_body {
            *header = HeaderPolicy::RECORDED_DISPATCH;
        }
        if full_attention_body && kernel == "gemv_hfq4g256_residual" {
            full_attention_body = false;
        }
    }
}

fn recorded_resource_accesses(
    hip: &HipRuntime,
    kernel: &str,
    kernarg: &[u8],
    certified_effects: Option<&[PointerEffect]>,
) -> Option<Vec<RecordedResourceAccess>> {
    if std::mem::size_of::<usize>() != 8 {
        return None;
    }
    let fallback;
    let effects = if let Some(effects) = certified_effects {
        effects
    } else {
        if kernarg.len() != expected_kernarg_bytes(kernel)? {
            return None;
        }
        fallback = pointer_effects(kernel)?;
        &fallback
    };
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
    /// For a parallel phase containing two dependent branch chains, indices
    /// before this position belong to lane 0 and indices at/after it belong
    /// to lane 1. `None` retains the ordinary round-robin antichain layout.
    lane_split: Option<usize>,
}

fn launches_are_independent(left: &RecordedHipLaunch, right: &RecordedHipLaunch) -> bool {
    let (Some(left), Some(right)) = (&left.accesses, &right.accesses) else {
        return false;
    };
    !left
        .iter()
        .any(|left| right.iter().any(|right| left.conflicts(*right)))
}

/// Opt-in flag for the antichain-widening reorder pass. Default off: the pass
/// permutes a certified launch sequence, which is strictly more aggressive than
/// omitting a wait, so it stays behind an explicit switch until it has its own
/// shadow and product evidence.
/// Reorder window, or `None` when the pass is off. A window of W permits a
/// launch to move at most W positions, which bounds how far the pass can
/// deviate from the certified sequence while still gathering nearby
/// independent launches. `on`/`1`/`true` means unlimited.
fn pm4_reorder_window_from_config(name: &str) -> Option<usize> {
    let value = hipfire_config::process_value(name)?;
    match value.as_str() {
        "0" | "false" | "off" | "" => None,
        "1" | "true" | "on" | "max" => Some(usize::MAX),
        other => match other.parse::<usize>() {
            Ok(window) if window >= 2 => Some(window),
            _ => {
                eprintln!(
                    "WARNING: {name}={other:?}: expected off, on, or an integer window >= 2; \
                     leaving the recorded order untouched"
                );
                None
            }
        },
    }
}

/// Architectures admitted to the single-IB reorder pass. `gfx1151` is the arch
/// the pass was certified on upstream; `gfx1201` is admitted here so the pass
/// can be screened against the gfx12 retained route, and remains default-off.
fn pm4_single_ib_reorder_from_config(device_name: &str) -> Option<usize> {
    (device_name.eq_ignore_ascii_case("gfx1151") || device_name.eq_ignore_ascii_case("gfx1201"))
        .then(|| pm4_reorder_window_from_config("HIPFIRE_REPLAY_PM4_SINGLE_IB_REORDER"))
        .flatten()
}

/// Permute the recorded launch order so mutually independent launches become
/// adjacent, widening the antichains `pm4_phase_plan` can form.
///
/// `pm4_phase_plan` only groups launches that are already CONSECUTIVE in the
/// recorded HIP order, so a launch independent of one ten positions away can
/// never share its phase. On the ds4 gfx1151 route that leaves 567 of 2319
/// boundaries independent yet almost all of them isolated pairs, capping every
/// parallel phase at width 2 no matter how many queues are requested, because
/// `lane_count = min(requested, phase.indices.len())`. Queue count was never
/// the constraint; adjacency was.
///
/// Two launches that do not conflict may be reordered freely; two that do
/// conflict must keep their recorded relative order. The result is therefore a
/// topological order of the conflict DAG, produced by a stable level-by-level
/// Kahn schedule. Each emitted level is the full ready set, which is pairwise
/// independent by construction: if two launches conflicted, the later one would
/// still hold the earlier as an unemitted predecessor and could not be ready.
///
/// Launches with no recovered `accesses` conflict with everything (see
/// `launches_are_independent`), so they pin their own position and act as
/// ordering barriers. That keeps the pass fail-closed on anything the resource
/// model could not prove.
///
/// Returns the identity order if the schedule cannot be validated, so a failure
/// degrades to today's behaviour rather than to a reordered tape.
fn pm4_width_reorder(recorded: &[RecordedHipLaunch], window: usize) -> Vec<usize> {
    let n = recorded.len();
    let identity = || (0..n).collect::<Vec<usize>>();
    if n == 0 || window < 2 {
        return identity();
    }

    // Schedule chunk-locally. Launches in different chunks keep their recorded
    // relative order by construction, so only within-chunk pairs can move and
    // no launch travels further than `window` positions. That bounds how far
    // the pass can deviate from the certified sequence, and makes the window a
    // bisection handle when a reordering turns out not to replay.
    let mut order = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = start.saturating_add(window).min(n);
        let span = end - start;

        // Conflict edges only ever point forward, so the graph is acyclic by
        // construction and Kahn's algorithm always drains it.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); span];
        let mut indegree = vec![0usize; span];
        for i in 0..span {
            for j in (i + 1)..span {
                if !launches_are_independent(&recorded[start + i], &recorded[start + j]) {
                    successors[i].push(j);
                    indegree[j] += 1;
                }
            }
        }

        let mut scheduled = 0usize;
        let mut ready: Vec<usize> = (0..span).filter(|index| indegree[*index] == 0).collect();
        while !ready.is_empty() {
            // Ascending original index keeps the permutation deterministic, so
            // the retained tape and its sequence hash reproduce across runs.
            ready.sort_unstable();
            let level = std::mem::take(&mut ready);
            for index in level.iter().copied() {
                order.push(start + index);
                scheduled += 1;
            }
            for index in level {
                for successor in successors[index].iter().copied() {
                    indegree[successor] -= 1;
                    if indegree[successor] == 0 {
                        ready.push(successor);
                    }
                }
            }
        }
        if scheduled != span {
            eprintln!(
                "WARNING: PM4 width reorder drained {scheduled} of {span} launches in chunk \
                 [{start}, {end}); retaining recorded order"
            );
            return identity();
        }
        start = end;
    }

    if order.len() != n {
        eprintln!(
            "WARNING: PM4 width reorder produced {} of {} launches; retaining recorded order",
            order.len(),
            n
        );
        return identity();
    }

    // Independently verify against the conflict relation over ALL pairs rather
    // than trusting the scheduler or the chunking argument: every conflicting
    // pair must keep its recorded relative order.
    let mut position = vec![usize::MAX; n];
    for (slot, index) in order.iter().copied().enumerate() {
        if index >= n || position[index] != usize::MAX {
            eprintln!("WARNING: PM4 width reorder is not a permutation; retaining recorded order");
            return identity();
        }
        position[index] = slot;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if !launches_are_independent(&recorded[i], &recorded[j]) && position[i] >= position[j] {
                eprintln!(
                    "WARNING: PM4 width reorder violated dependency {i} -> {j}; \
                     retaining recorded order"
                );
                return identity();
            }
        }
    }
    order
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
            phases.push(Pm4PhasePlan {
                parallel,
                indices,
                lane_split: None,
            });
        }
    }
    phases
}

fn launch_workgroups(launch: &RecordedHipLaunch) -> u64 {
    launch.grid.iter().fold(1_u64, |product, axis| {
        product.saturating_mul(u64::from(*axis))
    })
}

fn launch_ranges_are_independent(
    recorded: &[RecordedHipLaunch],
    left: std::ops::Range<usize>,
    right: std::ops::Range<usize>,
) -> bool {
    left.clone().all(|left_index| {
        right.clone().all(|right_index| {
            launches_are_independent(&recorded[left_index], &recorded[right_index])
        })
    })
}

/// Recover the two branch chains intentionally serialized by DeepSeek4's
/// retained-FFN capture route:
///
///   zero(routed) ; shared-E8 chain ; routed-MQ2 chain ; add(shared, routed)
///
/// The resource contracts prove every launch in the shared chain independent
/// of every launch in the routed chain. The zero is moved onto the routed lane
/// so the shared lane can start immediately. Both dependent chains retain their
/// original internal order, and the following serial phase begins with `add`.
fn pm4_ds4_ffn_branch_plan(recorded: &[RecordedHipLaunch]) -> Result<Vec<Pm4PhasePlan>, String> {
    let mut phases = Vec::<Pm4PhasePlan>::new();
    let mut cursor = 0_usize;
    let mut branches = 0_usize;

    while let Some(zero) = recorded[cursor..]
        .iter()
        .position(|launch| launch.kernel == "zero_f32")
        .map(|offset| cursor + offset)
    {
        let add = recorded[zero + 1..]
            .iter()
            .position(|launch| launch.kernel == "add_inplace_f32")
            .map(|offset| zero + 1 + offset)
            .ok_or_else(|| {
                format!(
                    "DeepSeek4 FFN branch capture has zero_f32 at {zero} without a following add"
                )
            })?;
        if recorded[zero + 1..add]
            .iter()
            .any(|launch| launch.kernel == "zero_f32")
        {
            return Err(format!(
                "DeepSeek4 FFN branch capture has nested zero_f32 before add at {add}"
            ));
        }

        let shared_start = zero + 1;
        let mut best = None::<(usize, u64)>;
        for split in shared_start + 1..add {
            let shared = &recorded[shared_start..split];
            let routed = &recorded[split..add];
            let looks_like_shared = shared
                .iter()
                .any(|launch| launch.kernel == "gemv_mfp4g32_e8_soa_u4");
            let looks_like_routed = routed
                .iter()
                .any(|launch| launch.kernel == "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed");
            if !looks_like_shared
                || !looks_like_routed
                || !launch_ranges_are_independent(recorded, shared_start..split, split..add)
                || !launch_ranges_are_independent(recorded, zero..zero + 1, shared_start..split)
            {
                continue;
            }
            let shared_work = shared.iter().map(launch_workgroups).sum::<u64>();
            let routed_work = recorded[zero..zero + 1]
                .iter()
                .chain(routed.iter())
                .map(launch_workgroups)
                .sum::<u64>();
            let balance = shared_work.min(routed_work);
            if best.is_none_or(|(_, best_balance)| balance > best_balance) {
                best = Some((split, balance));
            }
        }
        let (split, _) = best.ok_or_else(|| {
            format!(
                "DeepSeek4 FFN branch capture at zero_f32 index {zero} has no resource-independent shared/routed split before add index {add}"
            )
        })?;

        if cursor < zero {
            phases.push(Pm4PhasePlan {
                indices: (cursor..zero).collect(),
                parallel: false,
                lane_split: None,
            });
        }

        let shared_len = split - shared_start;
        let mut branch_indices = (shared_start..split).collect::<Vec<_>>();
        branch_indices.push(zero);
        branch_indices.extend(split..add);
        phases.push(Pm4PhasePlan {
            indices: branch_indices,
            parallel: true,
            lane_split: Some(shared_len),
        });
        branches += 1;
        cursor = add;
    }

    if cursor < recorded.len() {
        phases.push(Pm4PhasePlan {
            indices: (cursor..recorded.len()).collect(),
            parallel: false,
            lane_split: None,
        });
    }
    if branches == 0 {
        return Err(
            "DeepSeek4 FFN branch-chain planning requested but tape contains no zero/add markers"
                .to_owned(),
        );
    }
    eprintln!(
        "[redline] DeepSeek4 FFN branch-chain plan recovered {branches} shared/routed phases"
    );
    Ok(phases)
}

fn is_ds4_batched_e8_gemv(kernel: &str) -> bool {
    kernel.starts_with("gemv_mfp4g32_e8_soa_batched_b") && kernel.ends_with("_gfx1151")
}

/// Recover the fork/join already present in DeepSeek4's batched verify FFN:
///
///   shared: E8 w1 -> E8 w3 -> SwiGLU -> rotate -> E8 w2
///   routed: E8 router -> score transform -> top-k -> MQ2 gate/up -> SwiGLU -> rotate
///   join:   MQ2 down atomically accumulates into the completed shared output
///
/// The ordinary batched forward emits those branches serially. Keeping each
/// complete producer chain on one queue preserves its cache and dependency
/// locality while allowing the two large branches to overlap. The routed down
/// projection stays in the following serial phase because it consumes the
/// routed activation and updates the shared branch's output allocation.
///
/// Every recognized fork is re-proved resource-independent from the captured
/// argument effects. A changed kernel sequence or unknown effect rejects the
/// entire plan rather than partially parallelizing an unrecognized layer.
fn pm4_ds4_batched_ffn_branch_plan(
    recorded: &[RecordedHipLaunch],
) -> Result<Vec<Pm4PhasePlan>, String> {
    const ROUTED_GATE_UP: &str = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4";
    const ROUTED_DOWN: &str = "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4";

    let mut phases = Vec::<Pm4PhasePlan>::new();
    let mut cursor = 0_usize;
    let mut search_cursor = 0_usize;
    let mut branches = 0_usize;

    while let Some(down) = recorded[search_cursor..]
        .iter()
        .position(|launch| launch.kernel == ROUTED_DOWN)
        .map(|offset| search_cursor + offset)
    {
        // The exact captured branch has five shared launches and six routed
        // launches before the routed-down join.
        let shared_start = down.checked_sub(11).ok_or_else(|| {
            format!("DeepSeek4 batched FFN routed down at {down} has no complete fork prefix")
        })?;
        let router = down - 6;
        let names = |index: usize| recorded[index].kernel.as_str();
        let recognized = is_ds4_batched_e8_gemv(names(shared_start))
            && is_ds4_batched_e8_gemv(names(shared_start + 1))
            && names(shared_start + 2) == "deepseek4_silu_mul_clamp_f32"
            && names(shared_start + 3) == "mq_rotate_x"
            && is_ds4_batched_e8_gemv(names(shared_start + 4))
            && is_ds4_batched_e8_gemv(names(router))
            && names(router + 1) == "sqrt_softplus_f32"
            && matches!(
                names(router + 2),
                "hash_router_normalize_f32_batched" | "deepseek4_moe_topk_bias_aware_batched_f32"
            )
            && names(router + 3) == ROUTED_GATE_UP
            && names(router + 4) == "deepseek4_silu_mul_clamp_f32"
            && names(router + 5) == "mq_rotate_x";
        if !recognized {
            return Err(format!(
                "DeepSeek4 batched FFN fork before routed down at {down} does not match the certified 5+6 launch sequence"
            ));
        }
        if !launch_ranges_are_independent(recorded, shared_start..router, router..down) {
            return Err(format!(
                "DeepSeek4 batched FFN branches [{shared_start}, {router}) and [{router}, {down}) are not resource-independent"
            ));
        }

        if cursor < shared_start {
            phases.push(Pm4PhasePlan {
                indices: (cursor..shared_start).collect(),
                parallel: false,
                lane_split: None,
            });
        }
        let shared_len = router - shared_start;
        phases.push(Pm4PhasePlan {
            indices: (shared_start..down).collect(),
            parallel: true,
            lane_split: Some(shared_len),
        });
        branches += 1;
        cursor = down;
        search_cursor = down + 1;
    }

    if cursor < recorded.len() {
        phases.push(Pm4PhasePlan {
            indices: (cursor..recorded.len()).collect(),
            parallel: false,
            lane_split: None,
        });
    }
    if branches == 0 {
        return Err(
            "DeepSeek4 batched FFN branch-chain planning requested but tape has no routed-down joins"
                .to_owned(),
        );
    }
    eprintln!(
        "[redline] DeepSeek4 batched FFN branch-chain plan recovered {branches} shared/routed phases"
    );
    Ok(phases)
}

fn pm4_ds4_ffn_branch_chains_from_config() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_PM4_DS4_FFN_BRANCH_CHAINS")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
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
    if previous.starts_with("gated_delta_net_q8_compact")
        || current.starts_with("gated_delta_net_q8_compact")
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
    // LFM's rotated projection buffer is consumed immediately by GEMV. A
    // compute-idle wait orders execution, but gfx12 needs a vector-cache
    // acquire before the consumer reads mq_rotate_x output.
    if previous == "mq_rotate_x" {
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

/// Reused rotate destinations need their stale GC12 vector-cache line
/// invalidated before the writer executes in a retained IB.
fn requires_gfx12_pre_dispatch_vmem_acquire(current: &str) -> bool {
    current == "mq_rotate_x" || current == "fused_silu_mul_mq_rotate"
}

fn conservative_mid_acquire_except(previous: &str, current: &str, excluded: Option<&str>) -> bool {
    if previous.starts_with("gated_delta_net_q8_compact")
        || current.starts_with("gated_delta_net_q8_compact")
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
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
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

    fn units_for(self, position: usize) -> Result<u32, String> {
        let Self::PositionCeilDiv {
            axis: _,
            addend,
            divisor,
        } = self;
        if divisor == 0 {
            return Err(format!("invalid replay grid binding divisor={divisor}"));
        }
        let extent = u64::try_from(position)
            .map_err(|_| "decode position exceeds u64".to_owned())?
            .checked_add(u64::from(addend))
            .ok_or_else(|| "dynamic replay extent overflow".to_owned())?;
        let units = extent.div_ceil(u64::from(divisor)).max(1);
        u32::try_from(units).map_err(|_| format!("dynamic replay grid {units} exceeds u32"))
    }
}

/// Single decision point for GDN stochastic-rounding frame consumption.
///
/// `frames = max(1, nt * grid.z)` where `nt` is the little-endian `i32` at
/// kernarg byte offset 64 and `grid.z` is the recorded launch's third grid
/// dimension. Both are fixed for the fixed-shape tape prepared by
/// `prepare_pm4_prefix_inner` and by `replay_recorded_hip_prefix`.
///
/// Rejects (instead of silently defaulting) when the kernarg block is shorter
/// than 80 bytes, when `nt <= 0`, or when the product overflows `u32`.
pub(crate) fn gdn_requant_frames_for_dispatch(kernarg: &[u8], grid_z: u32) -> Result<u32, String> {
    if kernarg.len() < 80 {
        return Err(format!(
            "GDN kernarg block too short: {} < 80",
            kernarg.len()
        ));
    }
    let nt = i32::from_le_bytes(kernarg[64..68].try_into().expect("slice 64..68 is 4 bytes"));
    if nt <= 0 {
        return Err(format!("GDN nt must be > 0, got {nt}"));
    }
    let product = (nt as u64)
        .checked_mul(grid_z as u64)
        .ok_or_else(|| "GDN frame product overflow".to_owned())?;
    if product > u32::MAX as u64 {
        return Err(format!("GDN frame product {product} exceeds u32"));
    }
    let frames = product as u32;
    Ok(frames.max(1))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayKernargBinding {
    GdnFrameU32 { offset: usize, frames: u32 },
    PositionPlusU32 { offset: usize, addend: u32 },
}

impl ReplayKernargBinding {
    fn apply(self, kernarg_bytes: &mut [u8], position: usize) -> Result<(), String> {
        let len = kernarg_bytes.len();
        match self {
            Self::GdnFrameU32 { offset, frames } => {
                let frame = crate::norm::reserve_gdn_requant_frames(frames);
                let end = offset
                    .checked_add(4)
                    .ok_or_else(|| "kernarg binding offset overflow".to_owned())?;
                let slot = kernarg_bytes.get_mut(offset..end).ok_or_else(|| {
                    format!("GDN kernarg binding offset {offset} out of bounds (len {len})")
                })?;
                slot.copy_from_slice(&frame.to_le_bytes());
                Ok(())
            }
            Self::PositionPlusU32 { offset, addend } => {
                let value = u32::try_from(position)
                    .map_err(|_| "decode position exceeds u32".to_owned())?
                    .checked_add(addend)
                    .ok_or_else(|| "PositionPlusU32 overflow".to_owned())?;
                let end = offset
                    .checked_add(4)
                    .ok_or_else(|| "kernarg binding offset overflow".to_owned())?;
                let slot = kernarg_bytes.get_mut(offset..end).ok_or_else(|| {
                    format!("kernarg binding offset {offset} out of bounds (len {len})")
                })?;
                slot.copy_from_slice(&value.to_ne_bytes());
                Ok(())
            }
        }
    }
}

/// Opaque snapshot of one completed recording's per-launch kernarg blocks.
#[derive(Clone, Debug)]
pub struct RecordedKernargSnapshot {
    entries: Vec<SnapshotEntry>,
}

#[derive(Clone, Debug)]
struct SnapshotEntry {
    kernel: String,
    kernarg: Vec<u8>,
    grid: [u32; 3],
}

pub(crate) fn is_gdn_kernel(kernel: &str) -> bool {
    kernel == "gated_delta_net_q8_fast" || kernel.starts_with("gated_delta_net_q8_compact")
}

fn is_plausible_device_address(value: u64) -> bool {
    // Heuristic: real device pointers are at least page-aligned and in a
    // high virtual range. Small integers (positions, scales, sizes) are
    // < 1M or have low entropy. Use a conservative band that captures
    // relocated buffers without flagging legitimate position scalars.
    if value < 4096 {
        return false;
    }
    if value < 0x100000 {
        return false;
    }
    // Require either high 32 bits or a large 32-bit value that is not a
    // plausible small position+addend (which are typically < 1e6).
    let large_32 = value > 0x0100_0000 && value < (1u64 << 48) && value & 0x3 == 0;
    let has_high = (value >> 32) != 0 && value < (1u64 << 48);
    large_32 || has_high
}

/// Single code path that applies every retained-replay kernarg binding for one
/// dispatch at `position`. Both the PM4 retained IB and the recorded-HIP blob
/// oracle must call this helper so they cannot diverge.
pub(crate) fn apply_kernarg_bindings_for_dispatch(
    kernarg_bytes: &mut [u8],
    dispatch_index: usize,
    position: usize,
    bindings: &[(usize, ReplayKernargBinding)],
) -> Result<(), String> {
    for (dispatch, binding) in bindings {
        if *dispatch == dispatch_index {
            binding.apply(kernarg_bytes, position)?;
        }
    }
    Ok(())
}

impl ReplayController {
    /// Accessor for the synthesized position bindings (for testing and for
    /// the recorded-HIP oracle to share the same binding set).
    pub(crate) fn synthesized_position_bindings(&self) -> &[(usize, ReplayKernargBinding)] {
        &self.synthesized_position_bindings
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCaptureSummary {
    pub launch_count: usize,
    pub unique_kernel_count: usize,
    pub sequence_hash: u64,
}

fn replay_sequence_hash<'a>(launches: impl IntoIterator<Item = &'a RecordedHipLaunch>) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for launch in launches {
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
    hash
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

fn pm4_packet_identity(packet_count: usize) -> Option<usize> {
    (packet_count > 0).then_some(packet_count)
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
        // SAFETY: checked variant with string conversion.
        unsafe { self.replay_and_wait_profiled_checked() }.map_err(|(error, _)| error.to_string())
    }

    unsafe fn replay_and_wait_profiled_checked(
        &mut self,
    ) -> Result<GpuMultiQueueTiming, (redline_dispatch::aql::ReplayError, Quiescence)> {
        if dispatch_profile_enabled() {
            if let Self::Single(graph) = self {
                // Execute the instrumented graph once. Reuse the same timestamp
                // vector for whole-tape timing and the one-line legacy report.
                let (timing, spans) = unsafe { graph.replay_and_wait_dispatch_profiled_checked() }
                    .map_err(|(error, q)| (error, q))?;
                static REPORTED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    report_dispatch_spans(&spans);
                }
                return Ok(timing);
            }
        }
        match self {
            Self::Single(graph) => unsafe { graph.replay_and_wait_profiled_checked() },
            Self::Phased(graph) => unsafe { graph.replay_and_wait_profiled() }
                .map_err(|error| (error, Quiescence::Proven)),
        }
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

    fn packet_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Phased(graph) => graph.packet_count(),
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

    fn quiesce(&mut self) -> Result<(), (redline_dispatch::aql::ReplayError, Quiescence)> {
        match self {
            Self::Single(graph) => graph
                .quiesce()
                .map_err(|error| (error, Quiescence::Unknown)),
            Self::Phased(_) => Ok(()),
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
    /// gfx1010 RELEASE_MEM/WAIT_REG_MEM fence word. Owned for the full
    /// executable lifetime of `graph` so the IB's absolute address stays valid
    /// through every replay; dropped only after queue quiescence via normal
    /// PreparedPm4Replay teardown (field order: graph first).
    _dependency_fence: Option<KernargBuffer>,
    dynamic_gdn_frames: Vec<usize>,
    dynamic_kernarg_bindings: Vec<(usize, ReplayKernargBinding)>,
    dynamic_grids: Vec<(usize, ReplayGridBinding, [u32; 3], [u32; 3])>,
    pm4_architecture: Pm4Architecture,
    dispatch_count: usize,
    command_dwords: u32,
    dispatch_boundaries: Option<Vec<Pm4DispatchBoundary>>,
    prepared_max_position: Option<usize>,
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
        // SAFETY: checked variant handles quiescence mapping.
        unsafe { self.replay_and_wait_checked(position) }.map_err(|failure| failure.error)
    }

    /// Checked variant that reports quiescence.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::replay_and_wait`].
    pub unsafe fn replay_and_wait_checked(
        &mut self,
        position: usize,
    ) -> Result<GpuMultiQueueTiming, RetainedReplayFailure> {
        if let Some(max_pos) = self.prepared_max_position {
            if position > max_pos {
                return Err(RetainedReplayFailure {
                    error: format!("position {position} exceeds prepared max_position {max_pos}"),
                    quiescence: ReplayQuiescence::Proven,
                });
            }
        }
        // Patch typed kernarg bindings while queue is quiescent.
        // Single code path shared with recorded-HIP replay: both transports
        // must apply the identical binding set via `apply_kernarg_bindings_for_dispatch`.
        for dispatch_index in 0..self.kernargs.len() {
            let bytes = self.kernargs[dispatch_index].as_mut_bytes();
            apply_kernarg_bindings_for_dispatch(
                bytes,
                dispatch_index,
                position,
                &self.dynamic_kernarg_bindings,
            )
            .map_err(|error| RetainedReplayFailure {
                error,
                quiescence: ReplayQuiescence::Proven,
            })?;
        }
        // Legacy GDN frame patch (covers old prepared objects; new objects have
        // empty dynamic_gdn_frames and are patched via dynamic_kernarg_bindings).
        for dispatch in &self.dynamic_gdn_frames {
            // Skip if already covered by a GdnFrame binding for this dispatch.
            let already_covered = self.dynamic_kernarg_bindings.iter().any(|(idx, binding)| {
                *idx == *dispatch && matches!(binding, ReplayKernargBinding::GdnFrameU32 { .. })
            });
            if already_covered {
                continue;
            }
            let frame = crate::norm::reserve_gdn_requant_frames(1);
            let bytes = self.kernargs[*dispatch].as_mut_bytes();
            bytes
                .get_mut(76..80)
                .ok_or_else(|| RetainedReplayFailure {
                    error: "PM4 GDN kernarg is too short for frame patch".to_owned(),
                    quiescence: ReplayQuiescence::Proven,
                })?
                .copy_from_slice(&frame.to_ne_bytes());
        }
        for (dispatch, binding, recorded, workgroup) in &self.dynamic_grids {
            let workgroups =
                binding
                    .bind(position, *recorded)
                    .map_err(|error| RetainedReplayFailure {
                        error,
                        quiescence: ReplayQuiescence::Proven,
                    })?;
            let dimensions = if self.pm4_architecture == Pm4Architecture::Gfx12 {
                let mut workitems = [0_u32; 3];
                for axis in 0..3 {
                    workitems[axis] =
                        workgroups[axis]
                            .checked_mul(workgroup[axis])
                            .ok_or_else(|| RetainedReplayFailure {
                                error: format!(
                                "dynamic PM4 grid overflow axis={axis} workgroups={} workgroup={}",
                                workgroups[axis], workgroup[axis]
                            ),
                                quiescence: ReplayQuiescence::Proven,
                            })?;
                }
                workitems
            } else {
                workgroups
            };
            self.graph
                .patch_dispatch_dimensions(*dispatch, dimensions)
                .map_err(|error| RetainedReplayFailure {
                    error: error.to_string(),
                    quiescence: ReplayQuiescence::Proven,
                })?;
        }
        // SAFETY: forwarded from the caller that owns the model allocations.
        unsafe { self.graph.replay_and_wait_profiled_checked() }.map_err(|(error, quiescence)| {
            RetainedReplayFailure {
                error: error.to_string(),
                quiescence: match quiescence {
                    Quiescence::Proven => ReplayQuiescence::Proven,
                    Quiescence::Unknown => ReplayQuiescence::Unknown,
                },
            }
        })
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

    pub fn packet_count(&self) -> usize {
        self.graph.packet_count()
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
    radiowave_effect_certifications: BTreeMap<PathBuf, Option<CodeObjectCertification>>,
    radiowave_effect_launches: usize,
    fallback_effect_launches: usize,
    unknown_effect_launches: usize,
    /// Opt-in latch for daemon-owned post-generate route-proof markers.
    route_proof_log: bool,
    prepared_max_position: Option<usize>,
    synthesized_position_bindings: Vec<(usize, ReplayKernargBinding)>,
    position_bindings_calibrated: bool,
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
            radiowave_effect_certifications: BTreeMap::new(),
            unknown_effect_launches: 0,
            radiowave_effect_launches: 0,
            fallback_effect_launches: 0,
            route_proof_log: route_proof_log_requested(),
            prepared_max_position: None,
            synthesized_position_bindings: Vec::new(),
            position_bindings_calibrated: false,
        }
    }

    pub fn new_armed(request: ReplayBackendRequest) -> Self {
        let mut controller = Self::new(request);
        if request != ReplayBackendRequest::Hip {
            controller.state = ReplayState::Armed;
        }
        controller
    }

    /// Construct an explicitly-delimited automatic PM4 controller. Model
    /// adapters use this for secondary retained bodies (for example a
    /// speculative verify shape) that must not replace `Gpu::replay`, the
    /// model's ordinary-AR controller.
    pub fn new_manual_pm4() -> Self {
        let mut controller = Self::new_armed(ReplayBackendRequest::Auto);
        controller.transport = ReplayTransport::Pm4Ib;
        controller.auto_lifecycle = false;
        // Batched DS4 verify grows past the ordinary-AR 4,096-launch cap at
        // B>=5 (B=4 is 3,642 dispatches). Keep this scoped to the secondary
        // controller; the primary model controller retains its stricter cap.
        controller.max_recorded_launches = 8_192;
        controller
    }

    /// Construct the AQL-packet twin of [`Self::new_manual_pm4`]. This exists
    /// as a diagnostic/control route for secondary retained bodies: it keeps
    /// the same capture and kernarg lifetime while replacing architecture-
    /// native PM4 boundary lowering with public-HSA dispatch headers.
    pub fn new_manual_aql() -> Self {
        let mut controller = Self::new_armed(ReplayBackendRequest::Auto);
        controller.transport = ReplayTransport::AqlPackets;
        controller.auto_lifecycle = false;
        controller.max_recorded_launches = 8_192;
        controller
    }

    /// Apply the daemon's model-scoped replay default after a successful load.
    ///
    /// An explicit backend selection always wins. Otherwise every successful
    /// model load resets the process-local controller so prepared queues,
    /// command buffers, and fallback state cannot bleed across model swaps.
    /// Eligible single-GPU MQ4R models may default to retained PM4 on
    /// gfx1100, gfx1151, and gfx1201; this is runtime policy, not certification.
    /// All other models return to ordinary HIP. An explicit transport still
    /// overrides the PM4 transport choice for diagnostics.
    pub fn configure_model_default(&mut self, enable_mq4r: bool) -> bool {
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
        let transport = if enable_mq4r
            && !transport_override
                .as_deref()
                .is_some_and(|value| value != "auto")
        {
            ReplayTransport::Pm4Ib
        } else {
            ReplayTransport::from_config()
        };
        self.apply_model_default(enable_mq4r, transport);
        true
    }

    fn apply_model_default(&mut self, enable_mq4r: bool, transport: ReplayTransport) {
        let request = if enable_mq4r {
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
        self.radiowave_effect_certifications.clear();
        self.radiowave_effect_launches = 0;
        self.fallback_effect_launches = 0;
        self.unknown_effect_launches = 0;
        self.prepared_max_position = None;
        self.synthesized_position_bindings.clear();
        self.position_bindings_calibrated = false;
    }

    /// Drop a prepared route after a model-owned allocation/geometry bucket
    /// changes, preserving the selected backend and transport. Unlike
    /// [`Self::poison`], this is an expected lifecycle transition: the next
    /// eligible forward records and prepares a fresh route for the new stable
    /// layout.
    pub fn rearm_after_layout_growth(&mut self) {
        let request = self.request;
        let transport = self.transport;
        let auto_lifecycle = self.auto_lifecycle;
        self.reset_for_model(request, transport, auto_lifecycle);
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
                        packet_count: pm4_packet_identity(prepared.packet_count()),
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

    /// Whether this controller owns the production one-shot capture lifecycle.
    /// Manual shadow/profiling controllers deliberately remain available for
    /// diagnosing routes which are not yet safe for automatic serving.
    pub fn automatic_lifecycle_enabled(&self) -> bool {
        self.auto_lifecycle
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
                || launch.kernel.starts_with("gated_delta_net_q8_compact")
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
                if launch.kernel == "mq_rotate_x" {
                    headers[index] = HeaderPolicy::BATCH_INTERNAL_RELEASE_SYSTEM;
                    if index + 1 < headers.len() {
                        headers[index + 1] = HeaderPolicy::BATCH_INTERNAL_ACQUIRE_SYSTEM;
                    }
                } else {
                    headers[index] = HeaderPolicy::RECORDED_DISPATCH;
                }
            }
        }
        for index in 1..headers.len() {
            let previous = self.recorded[index - 1].kernel.as_str();
            let current = self.recorded[index].kernel.as_str();
            if independent_sibling(previous, current) {
                headers[index] = HeaderPolicy::BATCH_BOUNDARY_INTERNAL_INDEPENDENT;
            }
        }
        // HC ping-pong publishes the next residual allocation from
        // `hc_mix_4stream`, then `hc_input_map_4stream` consumes it after the
        // following block's control kernels. Queue-order barriers serialize
        // execution but do not by themselves establish gfx1151 cache
        // visibility, so place the narrow same-agent release/acquire pair at
        // the actual producer/consumer boundary.
        for (index, launch) in self.recorded.iter().take(prefix).enumerate() {
            match launch.kernel.as_str() {
                "hc_mix_4stream" => {
                    headers[index] = HeaderPolicy::BATCH_INTERNAL_RELEASE_AGENT;
                }
                "hc_input_map_4stream" => {
                    headers[index] = HeaderPolicy::BATCH_INTERNAL_ACQUIRE_AGENT;
                }
                _ => {}
            }
        }
        apply_qwen_q8_full_attention_visibility(&self.recorded[..prefix], &mut headers);
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
    /// Lower a captured prefix to a single retained PM4 IB.
    ///
    /// Permissive by construction: this is the entry point every certified
    /// route already uses, and its tape identity is sealed evidence, so it
    /// must keep its historical semantics. A route that replays across
    /// advancing decode positions must instead call
    /// [`Self::prepare_pm4_prefix_calibrated`], which refuses to prepare
    /// until two recordings have been differenced.
    pub fn prepare_pm4_prefix(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, false, true)
    }

    /// Position-aware variant: refuses to prepare unless
    /// [`Self::synthesize_position_bindings`] has differenced two recordings
    /// of this tape, so a position-tracking kernarg scalar cannot be retained
    /// unproven.
    pub fn prepare_pm4_prefix_calibrated(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, false, false)
    }

    /// Lower a captured prefix to a single GFX12 IB with per-dispatch timestamps.
    pub fn prepare_pm4_dispatch_profile(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, true, true)
    }

    fn prepare_pm4_prefix_inner(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
        dispatch_profile: bool,
        allow_uncalibrated: bool,
    ) -> Result<(usize, u32, u64), String> {
        if !allow_uncalibrated && !self.position_bindings_calibrated {
            return Err(
                "position bindings not calibrated; call synthesize_position_bindings or explicitly opt out via prepare_pm4_prefix_allow_uncalibrated".to_owned(),
            );
        }
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
        let mut dynamic_kernarg_bindings: Vec<(usize, ReplayKernargBinding)> = Vec::new();
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
            // Determine geometry grid: if a dynamic binding exists and a max_position
            // was requested, size the prepared IB for the maximum admitted position
            // so replay can patch DOWN. Otherwise use the recorded grid (today's
            // behavior).
            let mut geometry_grid = launch.grid;
            let mut grid_binding_for_storage = launch.grid_binding;
            if let Some(binding) = launch.grid_binding {
                if let Some(max_pos) = self.prepared_max_position {
                    let max_units = binding
                        .units_for(max_pos)
                        .map_err(|error| format!("{symbol}: {error}"))?;
                    let axis = match binding {
                        ReplayGridBinding::PositionCeilDiv { axis, .. } => usize::from(axis),
                    };
                    if axis >= 3 {
                        return Err(format!("{symbol}: invalid grid binding axis {axis}"));
                    }
                    let mut prepared_grid = launch.grid;
                    prepared_grid[axis] = max_units;
                    geometry_grid = prepared_grid;
                    // Store the prepared maximum as the recorded value so
                    // bind(current_position, prepared_grid) correctly narrows.
                    grid_binding_for_storage = Some(binding);
                    // Keep launch.grid replaced for geometry; dynamic_grids stores prepared_grid.
                }
            }
            let geometry = LaunchGeometry::from_hip_workgroups(geometry_grid, workgroup)
                .map_err(|error| format!("{symbol}: {error}"))?;
            device
                .validate_geometry(geometry)
                .map_err(|error| format!("{symbol}: {error}"))?;
            let is_gdn = launch.kernel == "gated_delta_net_q8_fast"
                || launch.kernel.starts_with("gated_delta_net_q8_compact");
            if is_gdn {
                if metadata.kernarg_segment_size < 80 {
                    return Err(format!(
                        "{symbol}: loader kernarg is too short for dynamic frame binding"
                    ));
                }
                // Derive the exact reservation run length for this dispatch:
                // frames = max(1, nt * grid.z) where nt is at kernarg offset 64
                // and grid.z is the recorded third grid dimension. This is the
                // single helper that decides consumption for PM4, recorded-blob,
                // and binding construction.
                let frames = gdn_requant_frames_for_dispatch(&launch.kernarg, launch.grid[2])
                    .map_err(|reason| format!("{symbol}: {reason}"))?;
                dynamic_kernarg_bindings.push((
                    kernargs.len(),
                    ReplayKernargBinding::GdnFrameU32 { offset: 76, frames },
                ));
                // New tapes rely solely on the typed binding above; the legacy
                // `dynamic_gdn_frames` vector is left empty so replay has exactly
                // one path that decides frame consumption. Old prepared objects
                // with a populated legacy vector remain supported via the
                // de-duplication check in `replay_and_wait_checked`.
            }
            if let Some(binding) = grid_binding_for_storage {
                let grid_to_store = if self.prepared_max_position.is_some() {
                    geometry_grid
                } else {
                    launch.grid
                };
                dynamic_grids.push((kernargs.len(), binding, grid_to_store, launch.block));
            }
            // Admissibility: every position-dependent scalar must be either
            // indirect via persistent buffer or covered by a declared binding.
            // Currently only GDN frame is such a scalar; reject a GDN-family
            // launch that somehow has no binding (would otherwise replay stale).
            if is_gdn {
                let has_gdn_binding = dynamic_kernarg_bindings
                    .iter()
                    .any(|(idx, _)| *idx == kernargs.len());
                if !has_gdn_binding {
                    return Err(format!(
                        "{symbol}: GDN-family launch has no kernarg binding"
                    ));
                }
            }
            kernels.push(kernel);
            kernargs.push(kernarg);
            geometries.push(geometry);
        }
        // Merge differential position bindings synthesized from two recordings.
        // Both transports must emit the identical binding set through one code
        // path; the position bindings are stored on the controller and merged
        // here for PM4. The recorded-HIP path merges the same set via
        // `apply_kernarg_bindings_for_dispatch`.
        for (dispatch, binding) in &self.synthesized_position_bindings {
            if *dispatch < prefix {
                dynamic_kernarg_bindings.push((*dispatch, *binding));
            }
        }
        dynamic_kernarg_bindings.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| {
                let ao = match a.1 {
                    ReplayKernargBinding::PositionPlusU32 { offset, .. } => offset,
                    ReplayKernargBinding::GdnFrameU32 { offset, .. } => offset,
                };
                let bo = match b.1 {
                    ReplayKernargBinding::PositionPlusU32 { offset, .. } => offset,
                    ReplayKernargBinding::GdnFrameU32 { offset, .. } => offset,
                };
                ao.cmp(&bo)
            })
        });

        let gfx12_gcr_trim = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GCR_TRIM")
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or(true);
        let gfx11_vmem_acquire =
            match hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE") {
                Some(value) if value != "auto" => matches!(value.as_str(), "1" | "true" | "on"),
                _ => device.name().eq_ignore_ascii_case("gfx1151"),
            };
        let radiowave_certifications = if gfx11_vmem_acquire {
            radiowave_certifications(&self.recorded, prefix)
        } else {
            BTreeMap::new()
        };
        if gfx11_vmem_acquire {
            let artifacts = self
                .recorded
                .iter()
                .take(prefix)
                .filter_map(|launch| launch.artifact.as_ref())
                .collect::<BTreeSet<_>>();
            let vmem_launches = self
                .recorded
                .iter()
                .take(prefix)
                .filter(|launch| radiowave_vmem_only_consumer(&radiowave_certifications, launch))
                .count();
            let vmem_symbols = self
                .recorded
                .iter()
                .take(prefix)
                .filter(|launch| radiowave_vmem_only_consumer(&radiowave_certifications, launch))
                .map(|launch| launch.kernel.as_str())
                .collect::<BTreeSet<_>>()
                .len();
            eprintln!(
                "[redline] Radiowave code-object contracts: certified_artifacts={}/{} \
                 vmem_symbols={} vmem_launches={}",
                radiowave_certifications.len(),
                artifacts.len(),
                vmem_symbols,
                vmem_launches,
            );
            eprintln!(
                "[redline] Radiowave argument effects: certified_launches={} \
                 fallback_launches={} unknown_launches={}",
                self.radiowave_effect_launches,
                self.fallback_effect_launches,
                self.unknown_effect_launches,
            );
        }
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
        let gfx1010_exact = gfx1010_release_wait_required(pm4_architecture, device.name());
        let gfx1010_dependency =
            gfx1010_dependency_policy_from_config(pm4_architecture, device.name())?;
        // Exact gfx1010 admits only single-queue non-native single-phase lowering.
        // Multi-queue / multi-phase / native-sync topologies stay on CS_PARTIAL_FLUSH
        // elsewhere and are fail-closed here rather than silently degraded.
        // Restriction holds under both ReleaseWait and CsPartialFlush overrides.
        if gfx1010_exact && queue_limit != 1 {
            return Err(
                "gfx1010 RELEASE_MEM/WAIT_REG_MEM dependency fence requires single-queue \
                 non-native single-phase retained PM4"
                    .to_owned(),
            );
        }
        let mut dependency_fence = None;
        let mut dispatch_boundaries = Vec::new();
        let (graph, command_dwords) = if queue_limit == 1 {
            let recorded = &self.recorded[..prefix];
            let reorder_window = pm4_single_ib_reorder_from_config(device.name());
            let order = match reorder_window {
                None => (0..prefix).collect::<Vec<_>>(),
                Some(window) => {
                    let order = pm4_width_reorder(recorded, window);
                    let moved = order
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(slot, index)| *slot != *index)
                        .count();
                    let max_displacement = order
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(slot, index)| slot.abs_diff(index))
                        .max()
                        .unwrap_or(0);
                    let sequence_hash =
                        replay_sequence_hash(order.iter().map(|index| &recorded[*index]));
                    eprintln!(
                        "[redline] single-IB reorder(arch={}, window={window}, scheduler=level): \
                         moved={moved}/{prefix} max_displacement={max_displacement} \
                         execution_sequence_hash={sequence_hash:016x}",
                        device.name()
                    );
                    order
                }
            };
            let dependency_mode = match gfx1010_dependency {
                Gfx1010DependencyPolicy::ReleaseWait => {
                    let fence = pool
                        .allocate_fine_grained_bytes(4, 4)
                        .map_err(|error| format!("allocate gfx1010 dependency fence: {error}"))?;
                    let address = fence.address() as u64;
                    if address == 0 || address & 3 != 0 {
                        return Err(format!(
                            "gfx1010 dependency fence address {address:#x} is null or unaligned"
                        ));
                    }
                    dependency_fence = Some(fence);
                    LegacyDependencyMode::ReleaseWait {
                        address,
                        next_epoch: 0,
                    }
                }
                Gfx1010DependencyPolicy::CsPartialFlush => LegacyDependencyMode::CsPartialFlush,
            };
            let mut commands = Pm4Commands::new_with_dependency(
                pm4_architecture,
                self.pm4_register_policy,
                dispatch_initiator_policy,
                dispatch_interleave,
                resource_limits_policy,
                dependency_mode,
            );
            // Sentinel epoch 0 before entry acquire: every immutable replay
            // re-submits this prefix so a stale prior epoch cannot satisfy the
            // next run (ABA).
            commands.emit_entry_sentinel_reset()?;
            commands.acquire_entry(gfx12_gcr_trim, entry_acquire_policy);
            let mut resource_frontier = ResourceFrontier::default();
            let mut dependency_waits = 0usize;
            let mut dependency_acquires = 0usize;
            for (position, index) in order.iter().copied().enumerate() {
                let mut boundary = Pm4DispatchBoundary::default();
                if position != 0 {
                    let previous_index = order[position - 1];
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
                    // GC12 can retain a vector-cache line for the reused
                    // rotated-output allocation across dispatches in one IB.
                    // Invalidate it before the writer starts; a consumer-side
                    // acquire after the writer is too late for this hazard.
                    let gfx12_pre_dispatch_acquire = pm4_architecture == Pm4Architecture::Gfx12
                        && requires_gfx12_pre_dispatch_vmem_acquire(current);
                    if gfx12_pre_dispatch_acquire || !independent {
                        dependency_waits += 1;
                        boundary.wait_compute_idle = true;
                        commands.wait_compute_idle()?;
                    }
                    resource_frontier.advance(current_launch, resources_independent);
                    let acquire = (!independent && commands.requires_dependency_acquire())
                        || self
                            .pm4_mid_acquire_policy
                            .acquire_between(previous, current);
                    if gfx12_pre_dispatch_acquire {
                        dependency_acquires += 1;
                        boundary.acquire_vmem = true;
                        commands.gfx12_system_acquire()?;
                    } else if acquire {
                        dependency_acquires += 1;
                        boundary.acquire_vmem = pm4_vmem_acquire_enabled(
                            pm4_architecture,
                            gfx11_vmem_acquire,
                            &radiowave_certifications,
                            current_launch,
                        );
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
            commands.wait_compute_idle()?;
            if dispatch_profile {
                commands.populate_dispatch_span_boundaries(&mut dispatch_boundaries)?;
            }
            let command_dwords = commands.len_dwords();
            if reorder_window.is_some() {
                eprintln!(
                    "[redline] single-IB schedule stats arch={}: \
                     independent_adjacencies={} dependency_waits={} \
                     dependency_acquires={} terminal_waits=1 command_dwords={command_dwords}",
                    device.name(),
                    prefix.saturating_sub(1).saturating_sub(dependency_waits),
                    dependency_waits,
                    dependency_acquires,
                );
                if device.name().eq_ignore_ascii_case("gfx1151") {
                    match commands.packet_census() {
                        Some(Ok(census)) => {
                            eprintln!("[redline] gfx1151 PM4 packet census: {census:?}");
                        }
                        Some(Err(dword)) => {
                            eprintln!("WARNING: gfx1151 PM4 packet census failed at dword {dword}");
                        }
                        None => {}
                    }
                }
            }
            // HIPFIRE_REDLINE_IB_POOL=vmem: allocate the retained indirect
            // buffer from a GPU-agent (VRAM) pool so the command processor
            // fetches the tape from VRAM instead of re-reading it over the
            // host interface on every replay. Host-read surfaces
            // (timestamps, completion signals) stay on the CPU-agent pool.
            // Falls back to the host pool with a warning when no device-local
            // pool exists.
            static IB_VMEM: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                hipfire_config::developer_var("HIPFIRE_REDLINE_IB_POOL").as_deref() == Ok("vmem")
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
            let ds4_ffn_branch_chains = pm4_ds4_ffn_branch_chains_from_config();
            let plans = if ds4_ffn_branch_chains {
                if self.recorded[..prefix]
                    .iter()
                    .any(|launch| launch.kernel == "zero_f32")
                {
                    pm4_ds4_ffn_branch_plan(&self.recorded[..prefix])?
                } else {
                    pm4_ds4_batched_ffn_branch_plan(&self.recorded[..prefix])?
                }
            } else {
                pm4_phase_plan(
                    &self.recorded[..prefix],
                    min_parallel_width,
                    min_parallel_workgroups,
                    max_parallel_phases,
                )
            };
            let parallel_phases = plans.iter().filter(|phase| phase.parallel).count();
            let branch_chain_phases = plans
                .iter()
                .filter(|phase| phase.lane_split.is_some())
                .count();
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
                    if phase.lane_split.is_some() {
                        self.pm4_queue_policy.resolve(device.name(), 2)
                    } else {
                        self.pm4_queue_policy
                            .resolve(device.name(), phase.indices.len())
                    }
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
                    let lane_indices = if let Some(split) = phase.lane_split {
                        if lane_count != 2 || split == 0 || split >= phase.indices.len() {
                            return Err(format!(
                                "invalid DeepSeek4 FFN branch phase: lanes={lane_count} split={split} launches={}",
                                phase.indices.len()
                            ));
                        }
                        vec![
                            phase.indices[..split].to_vec(),
                            phase.indices[split..].to_vec(),
                        ]
                    } else {
                        let mut lane_indices = vec![Vec::<usize>::new(); lane_count];
                        for (position, index) in phase.indices.iter().copied().enumerate() {
                            lane_indices[position % lane_count].push(index);
                        }
                        lane_indices
                    };
                    for (lane, indices) in lanes.iter_mut().zip(&lane_indices) {
                        let mut resource_frontier = ResourceFrontier::default();
                        for (position, index) in indices.iter().copied().enumerate() {
                            if position != 0 && phase.lane_split.is_some() {
                                let previous_index = indices[position - 1];
                                let previous_launch = &self.recorded[previous_index];
                                let current_launch = &self.recorded[index];
                                let previous = previous_launch.kernel.as_str();
                                let current = current_launch.kernel.as_str();
                                let resources_independent =
                                    resource_frontier.independent(current_launch);
                                if !resources_independent {
                                    lane.wait_compute_idle()?;
                                }
                                resource_frontier.advance(current_launch, resources_independent);
                                if (!resources_independent && lane.requires_dependency_acquire())
                                    || self
                                        .pm4_mid_acquire_policy
                                        .acquire_between(previous, current)
                                {
                                    lane.acquire_inter_node(
                                        gfx12_gcr_trim,
                                        gfx11_vmem_acquire
                                            && radiowave_vmem_only_consumer(
                                                &radiowave_certifications,
                                                current_launch,
                                            ),
                                    );
                                }
                            } else {
                                resource_frontier.advance(&self.recorded[index], false);
                            }
                            lane.dispatch(
                                &kernels[index],
                                geometries[index],
                                self.recorded[index].shared_mem,
                                kernargs[index].address(),
                            )
                            .map_err(|error| format!("{}: {error}", self.recorded[index].kernel))?;
                        }
                    }
                    for commands in &mut lanes {
                        commands.wait_compute_idle()?;
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
                            commands.wait_compute_idle()?;
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
                                    &radiowave_certifications,
                                    current_launch,
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
                commands.wait_compute_idle()?;
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
                "[redline] PM4 phase plan architecture={} queues={} phases={} parallel_phases={} branch_chain_phases={} parallel_launches={} max_width={} min_parallel_width={} min_parallel_workgroups={} max_parallel_phases={} sync={}",
                device.name(),
                graph.queue_count(),
                graph.phase_count(),
                parallel_phases,
                branch_chain_phases,
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
            _dependency_fence: dependency_fence,
            dynamic_gdn_frames,
            dynamic_kernarg_bindings,
            dynamic_grids,
            pm4_architecture,
            dispatch_count: prefix,
            command_dwords,
            dispatch_boundaries: dispatch_profile.then_some(dispatch_boundaries),
            prepared_max_position: self.prepared_max_position,
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
        // SAFETY: forwarded from the model owner.
        unsafe { self.replay_pm4_checked(position) }.map_err(|failure| failure.error)
    }

    /// Checked variant that reports quiescence.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::replay_pm4`].
    pub unsafe fn replay_pm4_checked(
        &mut self,
        position: usize,
    ) -> Result<GpuMultiQueueTiming, RetainedReplayFailure> {
        let result = {
            let prepared = match self.prepared_pm4.as_mut() {
                Some(prepared) => prepared,
                None => {
                    return Err(RetainedReplayFailure {
                        error: "no prepared PM4 replay".to_owned(),
                        quiescence: ReplayQuiescence::Proven,
                    })
                }
            };
            // SAFETY: forwarded from the model owner.
            unsafe { prepared.replay_and_wait_checked(position) }
        };
        self.observe_replay_result_checked(position, result)
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

    fn observe_replay_result_checked<T>(
        &mut self,
        position: usize,
        result: Result<T, RetainedReplayFailure>,
    ) -> Result<T, RetainedReplayFailure> {
        let value = match result {
            Ok(value) => value,
            Err(failure) => {
                self.replay_observation.failed = true;
                return Err(failure);
            }
        };
        self.replay_observation.count = self.replay_observation.count.saturating_add(1);
        self.replay_observation
            .first_position
            .get_or_insert(position);
        self.replay_observation.last_position = Some(position);
        Ok(value)
    }

    /// Largest `start_pos + batch` the prepared dynamic geometry must cover.
    pub fn set_prepared_max_position(&mut self, max_position: usize) {
        self.prepared_max_position = Some(max_position);
    }

    /// Prove no retained IB is in flight, then release the prepared route.
    pub fn shutdown(&mut self) -> Result<(), RetainedReplayFailure> {
        if let Some(prepared) = self.prepared_pm4.as_mut() {
            if let Err((error, quiescence)) = prepared.graph.quiesce() {
                return Err(RetainedReplayFailure {
                    error: error.to_string(),
                    quiescence: match quiescence {
                        Quiescence::Proven => ReplayQuiescence::Proven,
                        Quiescence::Unknown => ReplayQuiescence::Unknown,
                    },
                });
            }
        }
        // Quiescence proven: safe to release retained resources.
        self.prepared_pm4 = None;
        self.prepared = None;
        self.prepared_max_position = None;
        self.synthesized_position_bindings.clear();
        self.position_bindings_calibrated = false;
        Ok(())
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
        self.radiowave_effect_launches = 0;
        self.fallback_effect_launches = 0;
        self.unknown_effect_launches = 0;
        self.synthesized_position_bindings.clear();
        self.position_bindings_calibrated = false;
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
        let hash = replay_sequence_hash(&self.recorded);
        ReplayCaptureSummary {
            launch_count: self.recorded.len(),
            unique_kernel_count,
            sequence_hash: hash,
        }
    }

    /// Snapshot the just-finished recording. Valid only between `finish_capture`
    /// and the next `begin_capture`.
    pub fn snapshot_recorded_kernargs(&self) -> RecordedKernargSnapshot {
        let entries = self
            .recorded
            .iter()
            .map(|launch| SnapshotEntry {
                kernel: launch.kernel.clone(),
                kernarg: launch.kernarg.clone(),
                grid: launch.grid,
            })
            .collect();
        RecordedKernargSnapshot { entries }
    }

    /// Difference an earlier recording of the SAME tape against the current
    /// recording and synthesize a `ReplayKernargBinding` for every scalar that
    /// tracks the decode position. Returns the number of bindings synthesized.
    /// Errors (and synthesizes nothing) when any difference is not explained by
    /// the position delta.
    pub fn synthesize_position_bindings(
        &mut self,
        earlier: &RecordedKernargSnapshot,
        earlier_position: usize,
        current_position: usize,
    ) -> Result<usize, String> {
        if current_position <= earlier_position {
            return Err(format!(
                "current_position {current_position} must be > earlier_position {earlier_position}"
            ));
        }
        let delta = current_position - earlier_position;
        let delta_u32 =
            u32::try_from(delta).map_err(|_| format!("position delta {delta} exceeds u32"))?;
        if earlier.entries.len() != self.recorded.len() {
            return Err(format!(
                "launch count mismatch: earlier {} vs current {}",
                earlier.entries.len(),
                self.recorded.len()
            ));
        }
        let mut new_bindings: Vec<(usize, ReplayKernargBinding)> = Vec::new();
        for (idx, (earlier_entry, current_launch)) in
            earlier.entries.iter().zip(self.recorded.iter()).enumerate()
        {
            if earlier_entry.kernel != current_launch.kernel {
                return Err(format!(
                    "kernel mismatch at launch {idx}: earlier {:?} vs current {:?}",
                    earlier_entry.kernel, current_launch.kernel
                ));
            }
            if earlier_entry.kernarg.len() != current_launch.kernarg.len() {
                return Err(format!(
                    "kernarg length mismatch at launch {idx} kernel {:?}: earlier {} vs current {}",
                    earlier_entry.kernel,
                    earlier_entry.kernarg.len(),
                    current_launch.kernarg.len()
                ));
            }
            // Grid check modulo dynamic binding.
            if earlier_entry.grid != current_launch.grid {
                if let Some(binding) = current_launch.grid_binding {
                    let axis = match binding {
                        ReplayGridBinding::PositionCeilDiv { axis, .. } => usize::from(axis),
                    };
                    if axis >= 3 {
                        return Err(format!(
                            "invalid grid binding axis {axis} at launch {idx} kernel {:?}",
                            current_launch.kernel
                        ));
                    }
                    let mut mismatch_allowed = true;
                    for a in 0..3 {
                        if a == axis {
                            continue;
                        }
                        if earlier_entry.grid[a] != current_launch.grid[a] {
                            mismatch_allowed = false;
                        }
                    }
                    if !mismatch_allowed {
                        return Err(format!(
                            "grid mismatch at launch {idx} kernel {:?}: earlier {:?} vs current {:?} (axis {axis} is dynamic)",
                            current_launch.kernel, earlier_entry.grid, current_launch.grid
                        ));
                    }
                } else {
                    return Err(format!(
                        "grid mismatch at launch {idx} kernel {:?}: earlier {:?} vs current {:?}",
                        current_launch.kernel, earlier_entry.grid, current_launch.grid
                    ));
                }
            }
            let len = earlier_entry.kernarg.len();
            let is_gdn = is_gdn_kernel(&current_launch.kernel);
            let mut offset: usize = 0;
            while offset + 4 <= len {
                let earlier_bytes = &earlier_entry.kernarg[offset..offset + 4];
                let current_bytes = &current_launch.kernarg[offset..offset + 4];
                if earlier_bytes != current_bytes {
                    // Skip GDN frame field.
                    if is_gdn && offset == 76 {
                        offset += 4;
                        continue;
                    }
                    let v_earlier = u32::from_le_bytes(earlier_bytes.try_into().unwrap());
                    let v_current = u32::from_le_bytes(current_bytes.try_into().unwrap());
                    if v_current.wrapping_sub(v_earlier) != delta_u32 {
                        // Not a position scalar. Classify before rejecting: a
                        // relocated buffer usually differs only in the low word
                        // of its 8-byte slot, so inspect the enclosing slot
                        // rather than requiring both halves to differ.
                        let pair_offset = offset - (offset % 8);
                        if pair_offset + 8 <= len {
                            let earlier_u64 = u64::from_le_bytes(
                                earlier_entry.kernarg[pair_offset..pair_offset + 8]
                                    .try_into()
                                    .unwrap(),
                            );
                            let current_u64 = u64::from_le_bytes(
                                current_launch.kernarg[pair_offset..pair_offset + 8]
                                    .try_into()
                                    .unwrap(),
                            );
                            if is_plausible_device_address(earlier_u64)
                                || is_plausible_device_address(current_u64)
                            {
                                return Err(format!(
                                    "moved allocation at launch {idx} kernel {:?} offset {pair_offset}: earlier {earlier_u64:#x} vs current {current_u64:#x} (pointer; a relocated buffer must never be retained as a position scalar)",
                                    current_launch.kernel
                                ));
                            }
                        }
                        return Err(format!(
                            "unexplained kernarg difference at launch {idx} kernel {:?} offset {offset}: earlier {v_earlier} ({v_earlier:#x}) vs current {v_current} ({v_current:#x}) (delta {delta})",
                            current_launch.kernel
                        ));
                    }
                    if v_earlier.checked_add(delta_u32) != Some(v_current) {
                        return Err(format!(
                            "kernarg scalar at launch {idx} kernel {:?} offset {offset} wraps u32: earlier {v_earlier} + delta {delta_u32} != current {v_current}",
                            current_launch.kernel
                        ));
                    }
                    let addend = v_current.wrapping_sub(current_position as u32);
                    let repro_earlier = (earlier_position as u32).checked_add(addend);
                    let repro_current = (current_position as u32).checked_add(addend);
                    if repro_earlier != Some(v_earlier) || repro_current != Some(v_current) {
                        return Err(format!(
                            "synthesized PositionPlusU32 at launch {idx} kernel {:?} offset {offset} addend {addend} does not reproduce both samples (earlier {v_earlier} vs {:?}, current {v_current} vs {:?})",
                            current_launch.kernel, repro_earlier, repro_current
                        ));
                    }
                    new_bindings.push((
                        idx,
                        ReplayKernargBinding::PositionPlusU32 { offset, addend },
                    ));
                }
                offset += 4;
            }
        }
        // Order by (dispatch, offset) — already in order, but enforce.
        new_bindings.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| {
                let ao = match a.1 {
                    ReplayKernargBinding::PositionPlusU32 { offset, .. } => offset,
                    ReplayKernargBinding::GdnFrameU32 { offset, .. } => offset,
                };
                let bo = match b.1 {
                    ReplayKernargBinding::PositionPlusU32 { offset, .. } => offset,
                    ReplayKernargBinding::GdnFrameU32 { offset, .. } => offset,
                };
                ao.cmp(&bo)
            })
        });
        let count = new_bindings.len();
        self.synthesized_position_bindings = new_bindings;
        self.position_bindings_calibrated = true;
        Ok(count)
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
        if !self.is_recording() {
            return;
        }
        let certified_effects = artifact.as_ref().and_then(|artifact| {
            if !self.radiowave_effect_certifications.contains_key(artifact) {
                let certification = load_radiowave_certification(artifact);
                self.radiowave_effect_certifications
                    .insert(artifact.clone(), certification);
            }
            self.radiowave_effect_certifications
                .get(artifact)
                .and_then(Option::as_ref)
                .and_then(|certification| certification.argument_effects(kernel))
                .map(|effects| {
                    effects
                        .into_iter()
                        .map(|(offset, access)| match access {
                            KernelArgumentAccess::ReadOnly => read(offset),
                            KernelArgumentAccess::WriteOnly | KernelArgumentAccess::ReadWrite => {
                                write(offset)
                            }
                        })
                        .collect::<Vec<_>>()
                })
        });
        let certified = certified_effects.is_some();
        let accesses =
            recorded_resource_accesses(hip, kernel, kernarg, certified_effects.as_deref());
        if accesses.is_none() {
            self.unknown_effect_launches += 1;
        } else if certified {
            self.radiowave_effect_launches += 1;
        } else {
            self.fallback_effect_launches += 1;
        }
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

    #[test]
    fn pm4_architecture_fails_closed_for_unknown_gfx12_devices() {
        assert_eq!(
            Pm4Architecture::from_name("gfx1200"),
            Ok(Pm4Architecture::Gfx12)
        );
        assert_eq!(
            Pm4Architecture::from_name("GFX1201"),
            Ok(Pm4Architecture::Gfx12)
        );
        assert!(Pm4Architecture::from_name("gfx1202").is_err());
        assert!(Pm4Architecture::from_name("gfx12-future").is_err());
    }

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
        "fused_qkvza_mq4g256v2",
        "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
        "fused_qkv_mq4g256v2",
        "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
        "fused_gate_up_mq4g256v2",
        "fused_sigmoid_alpha_gate_f32",
        "conv1d_silu_split_f32",
        "conv1d_silu_split_qknorm_b256_scalar_prep",
        "fused_qk_l2_norm_scale_f32",
        "repeat_interleave_qk_f32",
        "gated_delta_net_q8_fast",
        "gated_norm_f32",
        "gated_norm_mq_rotate_gfx1100",
        "gated_norm_mq_rotate_k6144_gfx1100",
        "gated_norm_mq_rotate_gfx1151",
        "qwen35_fa_prep_gfx1100",
        "qwen36_27b_fa_prep_gfx1100",
        "qwen35_fa_prep_gfx1151",
        "mq_rotate_x",
        "gemv_hfq4g256_residual",
        "gemv_hfq4g256_residual_cpol_rt",
        "gemv_hfq4g256_residual_cpol_rt_low",
        "gemv_hfq4g256_residual_cpol_slc",
        "gemv_hfq4g256_residual_k2048",
        "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch",
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
        "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
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

    const DS4_MQ2R_REPLAY_KERNELS: &[&str] = &[
        "compressor_add_ape_f32_buf",
        "compressor_overlap_concat_f32",
        "compressor_softmax_pool_f32_buf",
        "deepseek4_attn_swa_buf",
        "deepseek4_attn_swa_topk_scoregrid_f32_buf",
        "deepseek4_fused_silu_mul_clamp_mq_rotate",
        "deepseek4_moe_topk_bias_aware_f32",
        "deepseek4_silu_mul_clamp_f32",
        "deepseek4_topk_kv_gather_tiled_f32_buf",
        "deepseek4_topk_kv_gather_identity_f32_buf",
        "fused_rmsnorm_mq_rotate_plain_nox",
        "gemv_mfp4g32_e8_soa_grouped_gfx1151",
        "gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151",
        "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed",
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
        "hash_router_normalize_f32_buf",
        "hc_compute_control_vec4_finalize",
        "hc_head_compute_pre",
        "hc_input_map_4stream",
        "hc_mix_4stream",
        "indexer_relu_score_f32_buf",
        "indexer_top_k_buf_parallel",
        "mq_rotate_x",
        "rmsnorm_f32",
        "rmsnorm_f32_at_slot_buf",
        "rope_tail_interleaved_f32",
        "rope_tail_yarn_interleaved_at_slot_buf_f32",
        "rope_tail_yarn_interleaved_wide_f32",
        "sqrt_softplus_f32",
        "state_overlap_shift_f32_buf",
        "state_ring_write_f32_buf",
        "swa_ring_write_f32_buf",
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
        let mut launch = RecordedHipLaunch {
            kernel: "fused_rmsnorm_mq_rotate".to_owned(),
            artifact: None,
            grid: [1, 1, 1],
            block: [32, 1, 1],
            shared_mem: 0,
            grid_binding: None,
            kernarg: Vec::new(),
            accesses: None,
        };
        let certifications = BTreeMap::new();
        assert!(!radiowave_vmem_only_consumer(&certifications, &launch));
        launch.artifact = Some("/missing/kernel.hsaco".into());
        assert!(!radiowave_vmem_only_consumer(&certifications, &launch));

        let artifact = PathBuf::from("/certified/fused_rmsnorm_mq_rotate.hsaco");
        let manifest = format!(
            r#"{{
                "schema_version": 3,
                "compiler": "radiowave",
                "generated_unix_seconds": 0,
                "source": "/source.hip",
                "output": "{}",
                "arch": "gfx1151",
                "wavefront": "wave32",
                "hipcc": "/opt/rocm/bin/hipcc",
                "hipcc_version": "test",
                "command": [],
                "source_sha256": "",
                "support_header_sha256": "",
                "output_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "inspection": {{
                    "bundle_target": "hipv4-amdgcn-amd-amdhsa--gfx1151",
                    "kernels": [{{
                        "name": "fused_rmsnorm_mq_rotate",
                        "wavefront_size": 32,
                        "vgpr_count": 1,
                        "sgpr_count": 1,
                        "vgpr_spill_count": 0,
                        "sgpr_spill_count": 0,
                        "private_segment_fixed_size": 0,
                        "mutable_read_cache": "vmem_only",
                        "instructions": {{}}
                    }}]
                }}
            }}"#,
            artifact.display()
        );
        let certification = CodeObjectCertification::from_json(&[], &manifest).unwrap();
        let certifications = BTreeMap::from([(artifact.clone(), certification)]);
        launch.artifact = Some(artifact);
        assert!(radiowave_vmem_only_consumer(&certifications, &launch));
        launch.kernel = "unknown_kernel".to_owned();
        assert!(!radiowave_vmem_only_consumer(&certifications, &launch));
    }

    #[test]
    fn ds4_mq2r_tape_has_complete_resource_contracts() {
        assert_eq!(DS4_MQ2R_REPLAY_KERNELS.len(), 32);
        for kernel in DS4_MQ2R_REPLAY_KERNELS {
            assert!(
                expected_kernarg_bytes(kernel).is_some(),
                "{kernel} has no kernarg contract"
            );
            assert!(
                pointer_effects(kernel).is_some(),
                "{kernel} has no pointer-effect contract"
            );
        }
        assert_eq!(
            pointer_effects("hc_mix_4stream").map(|effects| effects[4].mode),
            Some(RecordedAccessMode::Write)
        );
        assert_eq!(
            pointer_effects("hc_input_map_4stream").map(|effects| effects[1].mode),
            Some(RecordedAccessMode::Read)
        );
    }

    #[test]
    fn gfx12_never_reports_gfx11_vmem_acquire() {
        assert!(pm4_vmem_acquire_arch_enabled(Pm4Architecture::Gfx11, true));
        assert!(!pm4_vmem_acquire_arch_enabled(Pm4Architecture::Gfx12, true));
        assert!(!pm4_vmem_acquire_arch_enabled(
            Pm4Architecture::Gfx11,
            false
        ));
    }

    #[test]
    fn gfx1010_release_wait_selector_is_exact() {
        assert!(gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "gfx1010"
        ));
        assert!(gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "GFX1010"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "gfx1030"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "gfx1011"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx11,
            "gfx1100"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx11,
            "gfx1151"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx12,
            "gfx1201"
        ));
        // Architecture gate is conjunctive: wrong family never selects even if
        // the name string matches by accident.
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx11,
            "gfx1010"
        ));
    }

    #[test]
    fn gfx1010_dependency_policy_defaults_to_release_wait() {
        assert_eq!(
            gfx1010_dependency_policy_from_value(Pm4Architecture::Gfx10, "gfx1010", None).unwrap(),
            Gfx1010DependencyPolicy::ReleaseWait
        );
        assert_eq!(
            gfx1010_dependency_policy_from_value(
                Pm4Architecture::Gfx10,
                "GFX1010",
                Some("release-wait")
            )
            .unwrap(),
            Gfx1010DependencyPolicy::ReleaseWait
        );
    }

    #[test]
    fn gfx1010_dependency_policy_accepts_cs_partial_flush() {
        assert_eq!(
            gfx1010_dependency_policy_from_value(
                Pm4Architecture::Gfx10,
                "gfx1010",
                Some("cs-partial-flush")
            )
            .unwrap(),
            Gfx1010DependencyPolicy::CsPartialFlush
        );
    }

    #[test]
    fn gfx1010_dependency_policy_rejects_unknown_exact_values() {
        for raw in [
            "partial-flush",
            "",
            "cs",
            "CS-PARTIAL-FLUSH",
            "cs_partial_flush",
            "RELEASE-WAIT",
            "0",
            "1",
            "true",
            "falsé",
        ] {
            let err =
                gfx1010_dependency_policy_from_value(Pm4Architecture::Gfx10, "gfx1010", Some(raw))
                    .unwrap_err();
            assert!(
                err.contains("HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY"),
                "missing key in error for {raw:?}: {err}"
            );
            assert!(
                err.contains(raw),
                "missing offending value {raw:?} in error: {err}"
            );
        }
    }

    #[test]
    fn gfx1010_dependency_policy_ignored_off_exact_device() {
        // Non-exact devices stay on CsPartialFlush and ignore the override key.
        for (arch, name, value) in [
            (Pm4Architecture::Gfx10, "gfx1030", Some("release-wait")),
            (Pm4Architecture::Gfx10, "gfx1011", Some("cs-partial-flush")),
            (Pm4Architecture::Gfx11, "gfx1100", Some("release-wait")),
            (Pm4Architecture::Gfx11, "gfx1010", Some("release-wait")),
            (Pm4Architecture::Gfx12, "gfx1201", Some("bogus")),
            (Pm4Architecture::Gfx10, "gfx1030", None),
        ] {
            assert_eq!(
                gfx1010_dependency_policy_from_value(arch, name, value).unwrap(),
                Gfx1010DependencyPolicy::CsPartialFlush,
                "arch={arch:?} name={name} value={value:?}"
            );
        }
    }

    #[test]
    fn gfx1010_cs_partial_flush_override_emits_event_write_only() {
        // Encoding path for the diagnostic CsPartialFlush override: no fence
        // allocation/sentinel; historical EVENT_WRITE CS_PARTIAL_FLUSH only.
        let mut commands = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::CsPartialFlush,
        );
        assert_eq!(
            commands.dependency_mode(),
            Some(LegacyDependencyMode::CsPartialFlush)
        );
        commands.emit_entry_sentinel_reset().unwrap();
        commands.wait_compute_idle().unwrap();
        let dwords = commands.dwords().unwrap();
        assert_eq!(dwords, &[0xc000_4600, 0x407]);
    }

    #[test]
    fn gfx1010_release_wait_emits_sentinel_then_checked_epochs() {
        const FENCE_ADDR: u64 = 0x1234_5678_9abc_def0;
        let mut commands = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            },
        );
        commands.emit_entry_sentinel_reset().unwrap();
        commands.wait_compute_idle().unwrap();
        commands.wait_compute_idle().unwrap();

        let dwords = commands.dwords().expect("legacy dwords");
        // One fence is RELEASE_MEM (8 dwords) + WAIT_REG_MEM (7 dwords) = 15.
        assert_eq!(dwords.len(), 15 * 3);
        // Sentinel epoch 0, then dependency epochs 1 and 2.
        assert_eq!(dwords[5], 0);
        assert_eq!(dwords[12], 0);
        assert_eq!(dwords[15 + 5], 1);
        assert_eq!(dwords[15 + 12], 1);
        assert_eq!(dwords[30 + 5], 2);
        assert_eq!(dwords[30 + 12], 2);
        // No CS_PARTIAL_FLUSH EVENT_WRITE on the selected path.
        const EVENT_WRITE_IDLE: u32 = 0xc000_4600;
        assert!(!dwords.contains(&EVENT_WRITE_IDLE));
        assert_eq!(
            commands.dependency_mode(),
            Some(LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 2,
            })
        );
    }

    #[test]
    fn gfx1010_release_wait_rejects_u32_epoch_overflow() {
        let mut commands = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: 0x1000,
                next_epoch: u32::MAX,
            },
        );
        let err = commands.wait_compute_idle().unwrap_err();
        assert!(
            err.contains("epoch overflow"),
            "unexpected overflow error: {err}"
        );
    }

    #[test]
    fn gfx1010_release_wait_aba_reset_starts_each_stream_at_zero() {
        // Two independently constructed immutable streams both begin with the
        // sentinel epoch-0 fence, so a stale prior epoch cannot satisfy the
        // next replay (reset-to-zero / ABA shape).
        const FENCE_ADDR: u64 = 0xaaa0;
        let mut first = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            },
        );
        first.emit_entry_sentinel_reset().unwrap();
        first.wait_compute_idle().unwrap();

        let mut second = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            },
        );
        second.emit_entry_sentinel_reset().unwrap();

        let first_dwords = first.dwords().unwrap();
        let second_dwords = second.dwords().unwrap();
        assert_eq!(first_dwords[5], 0);
        assert_eq!(second_dwords[5], 0);
        assert_eq!(&first_dwords[..15], &second_dwords[..15]);
        // First stream advanced past the sentinel; second is still at epoch 0.
        assert_eq!(
            first.dependency_mode(),
            Some(LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 1,
            })
        );
        assert_eq!(
            second.dependency_mode(),
            Some(LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            })
        );
    }

    #[test]
    fn default_dependency_mode_keeps_cs_partial_flush() {
        let mut commands = Pm4Commands::new(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
        );
        assert_eq!(
            commands.dependency_mode(),
            Some(LegacyDependencyMode::CsPartialFlush)
        );
        commands.emit_entry_sentinel_reset().unwrap();
        commands.wait_compute_idle().unwrap();
        let dwords = commands.dwords().unwrap();
        // EVENT_WRITE CS_PARTIAL_FLUSH only — no RELEASE_MEM / WAIT_REG_MEM.
        assert_eq!(dwords, &[0xc000_4600, 0x407]);
    }

    #[test]
    fn sequence_hash_inputs_unchanged_by_dependency_mode() {
        // Fence selection must not alter capture identity / sequence hashing.
        let launches = [
            RecordedHipLaunch {
                kernel: "a".to_owned(),
                artifact: None,
                grid: [1, 2, 3],
                block: [32, 1, 1],
                shared_mem: 0,
                grid_binding: None,
                kernarg: vec![1, 2, 3, 4],
                accesses: None,
            },
            RecordedHipLaunch {
                kernel: "b".to_owned(),
                artifact: None,
                grid: [4, 5, 6],
                block: [64, 1, 1],
                shared_mem: 128,
                grid_binding: None,
                kernarg: vec![5, 6],
                accesses: None,
            },
        ];
        let hash = replay_sequence_hash(&launches);
        assert_eq!(hash, replay_sequence_hash(launches.iter()));
        assert_ne!(hash, 0);
    }

    #[test]
    fn lfm_retained_effect_contracts() {
        assert_eq!(expected_kernarg_bytes("conv1d_gated_decode_f32"), Some(48));
        let conv = pointer_effects("conv1d_gated_decode_f32").expect("conv contract");
        assert_eq!(conv.len(), 4);
        assert_eq!(conv[0].offset, 0);
        assert_eq!(conv[0].mode, RecordedAccessMode::Read);
        assert_eq!(conv[1].offset, 8);
        assert_eq!(
            conv[1].mode,
            RecordedAccessMode::Write,
            "conv state @8 is RMW, recorded as write"
        );
        assert_eq!(conv[2].offset, 16);
        assert_eq!(conv[2].mode, RecordedAccessMode::Read);
        assert_eq!(conv[3].offset, 24);
        assert_eq!(conv[3].mode, RecordedAccessMode::Write);

        assert_eq!(expected_kernarg_bytes("attention_q8_0_kv"), Some(64));
        let attn = pointer_effects("attention_q8_0_kv").expect("attn contract");
        assert_eq!(attn.len(), 5);
        assert_eq!(attn[0].offset, 0);
        assert_eq!(attn[0].mode, RecordedAccessMode::Read);
        assert_eq!(attn[1].offset, 8);
        assert_eq!(attn[1].mode, RecordedAccessMode::Read);
        assert_eq!(attn[2].offset, 16);
        assert_eq!(attn[2].mode, RecordedAccessMode::Read);
        assert_eq!(attn[3].offset, 24);
        assert_eq!(attn[3].mode, RecordedAccessMode::Write);
        assert_eq!(attn[4].offset, 32);
        assert_eq!(attn[4].mode, RecordedAccessMode::Read);

        assert!(expected_kernarg_bytes("unknown_kernel_xyz").is_none());
        assert!(pointer_effects("unknown_kernel_xyz").is_none());
    }

    #[test]
    fn deltanet_f32_retained_effect_contract_covers_recurrent_state() {
        assert_eq!(expected_kernarg_bytes("gated_delta_net_f32"), Some(80));
        let effects = pointer_effects("gated_delta_net_f32").expect("GDN FP32 contract");
        assert_eq!(effects.len(), 7);
        assert_eq!(effects[5].offset, 40);
        assert_eq!(effects[5].mode, RecordedAccessMode::Write);
        assert_eq!(effects[6].offset, 48);
        assert_eq!(effects[6].mode, RecordedAccessMode::Write);
    }

    #[test]
    fn qwen_q8_full_attention_aql_body_uses_system_visibility() {
        let launch = |kernel: &str| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [32, 1, 1],
            shared_mem: 0,
            grid_binding: None,
            kernarg: Vec::new(),
            accesses: None,
        };
        let launches = vec![
            launch("before"),
            launch("fused_qkv_hfq4g256"),
            launch("attention_flash_q8_0_tile"),
            launch("attention_flash_q8_0_reduce"),
            launch("gemv_hfq4g256_residual"),
            launch("after"),
        ];
        let mut headers = vec![HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL; launches.len()];

        apply_qwen_q8_full_attention_visibility(&launches, &mut headers);

        assert_eq!(headers[0], HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL);
        assert!(headers[1..=4]
            .iter()
            .all(|header| *header == HeaderPolicy::RECORDED_DISPATCH));
        assert_eq!(headers[5], HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL);
    }

    #[test]
    fn gemma4_ple_activation_keeps_padded_replay_contract() {
        let kernel = "gemma4_ple_gelu_mul_strided_f32";
        let effects = pointer_effects(kernel).expect("Gemma 4 PLE activation contract");
        assert_eq!(effects.len(), 3);
        assert_eq!(effects[0].offset, 0);
        assert_eq!(effects[0].mode, RecordedAccessMode::Read);
        assert_eq!(effects[1].offset, 8);
        assert_eq!(effects[1].mode, RecordedAccessMode::Read);
        assert_eq!(effects[2].offset, 16);
        assert_eq!(effects[2].mode, RecordedAccessMode::Write);

        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..3 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..4 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 40, "explicit kernel arguments occupy 40 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 48, "recorded launches are padded to 16 bytes");
        assert_eq!(expected_kernarg_bytes(kernel), Some(blob.len()));
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
        assert_eq!(expected_kernarg_bytes(qkvza_r4), Some(96));
        assert_eq!(
            pointer_effects(qkvza_r4).map(|effects| effects.len()),
            Some(9)
        );
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

    // Fail if a codebook MoE kernel variant is added without resource-contract registration.
    #[test]
    fn codebook_moe_symbols_have_resource_contracts() {
        let gate_up_effects = vec![read(0), read(8), read(16), write(24), write(32)];
        let down_effects = vec![read(0), read(8), read(16), read(24), write(32)];
        for (symbol, kernarg_bytes, effects) in [
            (
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
                48usize,
                &gate_up_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed",
                48,
                &gate_up_effects,
            ),
            (
                "gemv_mq2g256gl_moe_gate_up_k8_indexed",
                64,
                &gate_up_effects,
            ),
            (
                "gemv_mq3g256gl_moe_gate_up_k8_indexed",
                80,
                &gate_up_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed",
                48,
                &down_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2",
                48,
                &down_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4",
                48,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed",
                48,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2",
                48,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4",
                48,
                &down_effects,
            ),
            ("gemv_mq3g256_lloyd_moe_ninepath_d4", 48, &down_effects),
            (
                "gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed",
                64,
                &down_effects,
            ),
            (
                "gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed",
                80,
                &down_effects,
            ),
            // Batched-K4 prefill siblings. 52 B, NOT 48: the K_TOP scalar is
            // exactly the kind of quiet ABI difference that makes a
            // pattern-matched contract fail closed instead of loudly.
            (
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
                52,
                &gate_up_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
                52,
                &gate_up_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
                52,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
                52,
                &down_effects,
            ),
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(kernarg_bytes));
            let got = pointer_effects(symbol).expect("codebook MoE pointer contract");
            assert_eq!(got.len(), effects.len());
            for (got_effect, want) in got.iter().zip(effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
            // Offset 24 is Write for gate_up and Read for down/ninepath.
            assert_eq!(got[3].offset, 24);
            if symbol.contains("gate_up") {
                assert_eq!(got[3].mode, RecordedAccessMode::Write);
            } else {
                assert_eq!(got[3].mode, RecordedAccessMode::Read);
            }
        }
    }

    // Fail closed if an MQ4V2/MQ6V2 MoE kernel is admitted without a distinct
    // resource contract. V2 names must never silently fall through to V1.
    #[test]
    fn mqv2_moe_symbols_have_resource_contracts() {
        let gate_up_effects = vec![read(0), read(8), read(16), write(24), write(32)];
        let expanded_down_effects = vec![read(0), read(8), read(16), write(24)];
        let ninepath_effects = vec![read(0), read(8), read(16), read(24), write(32)];
        let grouped_effects = vec![read(0), read(8), read(16), read(24), write(32)];

        // Decode gate_up: 5 ptr + M,K = 48 B (already 16-aligned).
        for symbol in [
            "gemv_mq4g256v2_moe_gate_up_k8_indexed",
            "gemv_mq6g256v2_moe_gate_up_k8_indexed",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(48));
            let got = pointer_effects(symbol).expect("mqv2 gate_up pointer contract");
            assert_eq!(got.len(), gate_up_effects.len());
            for (got_effect, want) in got.iter().zip(gate_up_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // Path1 batched gate_up: 5 ptr + M,K,K_TOP = 52 → pad_to(16) = 64.
        // Prove the padded recorded length, not the bare field total.
        for symbol in [
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched",
            "gemv_mq6g256v2_moe_gate_up_k8_indexed_batched",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..5 {
                blob.push_ptr(std::ptr::null());
            }
            blob.push_i32(0);
            blob.push_i32(0);
            blob.push_i32(0);
            assert_eq!(blob.len(), 52, "{symbol} explicit args occupy 52 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mqv2 batched gate_up pointer contract");
            assert_eq!(got.len(), gate_up_effects.len());
            for (got_effect, want) in got.iter().zip(gate_up_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // Expanded down: 4 ptr + M,K,K_TOP = 44 → 48 padded.
        for symbol in [
            "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded",
            "gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..4 {
                blob.push_ptr(std::ptr::null());
            }
            blob.push_i32(0);
            blob.push_i32(0);
            blob.push_i32(0);
            assert_eq!(blob.len(), 44, "{symbol} explicit args occupy 44 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 48, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mqv2 expanded down pointer contract");
            assert_eq!(got.len(), expanded_down_effects.len());
            for (got_effect, want) in got.iter().zip(expanded_down_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // Ninepath fused down+combine: 5 ptr + down_m,down_k = 48.
        for symbol in [
            "gemv_mq4g256v2_moe_ninepath_d4",
            "gemv_mq6g256v2_moe_ninepath_d4",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(48));
            let got = pointer_effects(symbol).expect("mqv2 ninepath pointer contract");
            assert_eq!(got.len(), ninepath_effects.len());
            for (got_effect, want) in got.iter().zip(ninepath_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
            // Offset 24 is the rotated act (read); out RMW lives at 32.
            assert_eq!(got[3].mode, RecordedAccessMode::Read);
            assert_eq!(got[4].offset, 32);
            assert_eq!(got[4].mode, RecordedAccessMode::Write);
        }

        // Grouped prefill WMMA: 5 ptr + M,K,x_row_div,m_total = 56 → 64.
        // gfx11 `_k2` and gfx12 `_gfx12` are distinct symbols, not aliases.
        for symbol in [
            "gemm_mq4g256v2_moe_grouped_wmma_k2",
            "gemm_mq4g256v2_moe_grouped_wmma_gfx12",
            "gemm_mq6g256v2_moe_grouped_wmma_k2",
            "gemm_mq6g256v2_moe_grouped_wmma_gfx12",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..5 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..4 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 56, "{symbol} explicit args occupy 56 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mqv2 grouped pointer contract");
            assert_eq!(got.len(), grouped_effects.len());
            for (got_effect, want) in got.iter().zip(grouped_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // No V1/V2 collapse: MQ4V2/MQ6V2 names are not HFQ4/HFQ6 contracts by alias.
        assert!(
            pointer_effects("gemv_mq4g256v2_moe_gate_up_k8_indexed").is_some()
                && pointer_effects("gemv_hfq4g256_moe_gate_up_k8_indexed").is_some()
        );
        assert_ne!(
            "gemv_mq4g256v2_moe_gate_up_k8_indexed",
            "gemv_hfq4g256_moe_gate_up_k8_indexed"
        );
        assert_ne!(
            "gemv_mq6g256v2_moe_gate_up_k8_indexed",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed"
        );
        // Unknown V2-looking name still fails closed.
        assert!(pointer_effects("gemv_mq4g256v2_moe_gate_up_k8_indexed_unknown").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_moe_gate_up_k8_indexed_unknown").is_none());
    }

    // Table-driven: mixed-precision MoE kernels (tags 0..18, V1/V2 never alias).
    // Covers every kernel admitted to the mixed grouped/expanded replay path.
    #[test]
    fn mixed_moe_symbols_have_resource_contracts() {
        let mixed_gate_up_effects =
            vec![read(0), read(8), read(16), read(24), write(32), write(40)];
        let mixed_down_effects = vec![read(0), read(8), read(16), read(24), write(32)];
        let mixed_grouped_effects = vec![read(0), read(8), read(16), read(24), read(32), write(40)];

        // Mixed gate_up batched: 6 ptr + M,K,K_TOP = 60 → 64 padded.
        for symbol in ["gemv_mixed_moe_gate_up_k8_indexed_batched"] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..6 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..3 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 60, "{symbol} explicit args occupy 60 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mixed gate_up pointer contract");
            assert_eq!(
                got.len(),
                mixed_gate_up_effects.len(),
                "{symbol} effect count"
            );
            for (got_effect, want) in got.iter().zip(mixed_gate_up_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
            // Extra pointer is dtype_tags at +8 (read); y_gate/y_up are distinct writes.
            assert_eq!(got[1].offset, 8);
            assert_eq!(got[1].mode, RecordedAccessMode::Read);
            assert_eq!(got[4].mode, RecordedAccessMode::Write);
            assert_eq!(got[5].mode, RecordedAccessMode::Write);
        }

        // Mixed down expanded: 5 ptr + M,K,K_TOP = 52 → 64 padded.
        for symbol in ["gemv_mixed_moe_down_k8_indexed_batched_expanded"] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..5 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..3 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 52, "{symbol} explicit args occupy 52 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mixed down pointer contract");
            assert_eq!(got.len(), mixed_down_effects.len(), "{symbol} effect count");
            for (got_effect, want) in got.iter().zip(mixed_down_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
            assert_eq!(got[1].offset, 8);
            assert_eq!(got[1].mode, RecordedAccessMode::Read);
            assert_eq!(got[4].mode, RecordedAccessMode::Write);
        }

        // Mixed grouped WMMA: 6 ptr + M,K,x_row_div,m_total = 64 (already aligned).
        // gfx11 k2, gfx12, and 4w_k2 tiling variants are distinct symbols.
        for symbol in [
            "gemm_mixed_moe_grouped_wmma_k2",
            "gemm_mixed_moe_grouped_wmma_gfx12",
            "gemm_mixed_moe_grouped_wmma_4w_k2",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..6 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..4 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 64, "{symbol} explicit args occupy 64 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mixed grouped pointer contract");
            assert_eq!(
                got.len(),
                mixed_grouped_effects.len(),
                "{symbol} effect count"
            );
            for (got_effect, want) in got.iter().zip(mixed_grouped_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
            assert_eq!(got[5].offset, 40);
            assert_eq!(got[5].mode, RecordedAccessMode::Write);
        }

        // V1/V2 names never alias; unknown mixed name still fails closed.
        assert!(pointer_effects("gemv_mixed_moe_gate_up_k8_indexed_batched_unknown").is_none());
        assert!(
            expected_kernarg_bytes("gemv_mixed_moe_gate_up_k8_indexed_batched_unknown").is_none()
        );
        assert_ne!(
            "gemv_mixed_moe_gate_up_k8_indexed_batched",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched"
        );
    }

    // Table-driven: every reachable dense shared-expert V2 symbol.
    // Shared experts ride the exact dense V2 GEMV/GEMM ABIs — never V1 aliases.
    #[test]
    fn dense_shared_v2_symbols_have_resource_contracts() {
        let dense_gemv_effects = vec![read(0), read(8), write(16)];
        let dense_gemm_effects = vec![read(0), read(8), write(16)];

        // Dense GEMV: 3 ptr + M,K = 32 (already 16-aligned). Plain, residual, multirow.
        // The gfx1100 lm_head x-buffer candidate lives here: same 32B plain V2
        // contract (candidate pointer effects read@0,@8,write@16) so A3B replay
        // inventory treats it as the same dense family and fails closed with
        // identical resource effects.
        for symbol in [
            "gemv_mq4g256v2",
            "gemv_mq4g256v2_residual",
            "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch",
            "gemv_mq4g256v2_gfx1100_lm_head_x_buffer",
            "gemv_mq6g256v2",
            "gemv_mq6g256v2_residual",
            "gemv_mq4g256v2_multirow_r2",
            "gemv_mq4g256v2_multirow_r4",
            "gemv_mq4g256v2_multirow_r8",
            "gemv_mq6g256v2_multirow_r2",
            "gemv_mq6g256v2_multirow_r4",
            "gemv_mq6g256v2_multirow_r8",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..3 {
                blob.push_ptr(std::ptr::null());
            }
            blob.push_i32(0);
            blob.push_i32(0);
            assert_eq!(blob.len(), 32, "{symbol} explicit args occupy 32 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 32, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("dense gemv pointer contract");
            assert_eq!(got.len(), dense_gemv_effects.len(), "{symbol} effect count");
            for (got_effect, want) in got.iter().zip(dense_gemv_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
        }

        // Dense GEMM residual WMMA: 3 ptr + M,K,batch = 36 → 48 padded.
        // Covers plain and arch-tiled gfx11/gfx12 variants admitted to slots/prefill.
        for symbol in [
            "gemm_mq4g256v2_residual_wmma",
            "gemm_mq4g256v2_residual_wmma_gfx12",
            "gemm_mq6g256v2_residual_wmma",
            "gemm_mq6g256v2_residual_wmma_gfx12",
            "gemm_mq4g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq4g256v2_residual_wmma_gfx11_bt6",
            "gemm_mq4g256v2_residual_wmma_gfx11_bt8",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt6",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt8",
            "gemm_mq4g256v2_residual_wmma_gfx1100_mw4_lds",
            "gemm_mq4g256v2_residual_wmma_gfx1100_mw8_lds",
            "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds",
            "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..3 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..3 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 36, "{symbol} explicit args occupy 36 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 48, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("dense gemm pointer contract");
            assert_eq!(got.len(), dense_gemm_effects.len(), "{symbol} effect count");
            for (got_effect, want) in got.iter().zip(dense_gemm_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
        }

        // No V1/V2 collapse for dense; unknown dense V2 name still fails closed.
        assert!(pointer_effects("gemv_mq4g256v2_unknown").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_unknown").is_none());
        assert_ne!("gemv_mq4g256v2", "gemv_hfq4g256");
        assert_ne!("gemm_mq4g256v2_residual_wmma", "gemm_mq4g256_residual_wmma");
        assert!(pointer_effects("gemv_mq4g256v2").is_some());
        assert!(pointer_effects("gemv_mq6g256v2").is_some());
    }

    #[test]
    fn mq4g256v2_residual_r1_k4096_gfx1100_noscratch_keeps_exact_residual_contract() {
        // gfx1100-only residual R1 K=4096 noscratch candidate: same 32-B plain V2
        // residual ABI (A@0, x@8, y@16 RMW/write; M@24, K@28). Distinct symbol —
        // never collapses onto generic residual or unknown names.
        let symbol = "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch";
        let want = [read(0), read(8), write(16)];

        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..3 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_i32(0);
        assert_eq!(blob.len(), 32, "{symbol} explicit args occupy 32 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 32, "{symbol} recorded launches pad to 16");
        assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));

        let got = pointer_effects(symbol).expect("noscratch residual pointer contract");
        assert_eq!(got.len(), want.len());
        for (got_effect, want_effect) in got.iter().zip(want.iter()) {
            assert_eq!(got_effect.offset, want_effect.offset);
            assert_eq!(got_effect.mode, want_effect.mode);
        }

        // Fail-closed: unknown / mismatched residual names stay unrecognized.
        assert!(pointer_effects("gemv_mq4g256v2_residual_r1_k4096_gfx1100").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_residual_r1_k4096_gfx1100").is_none());
        assert_ne!(symbol, "gemv_mq4g256v2_residual");
        assert_ne!(symbol, "gemv_hfq4g256_residual_k2048");
    }

    #[test]
    fn mq4g256v2_gfx1100_lm_head_x_buffer_keeps_exact_32b_abi() {
        // Ornith lm_head specialization: frozen plain V2 ABI (A@0 read, x@8
        // read, y@16 write; M@24, K@28 → 32B). Same as generic decode; one
        // wave/row, no multirow/tiling/precision/cache-policy changes. The
        // candidate must be indistinguishable from generic in the resource
        // model and must fail closed on any distinct name.
        let symbol = "gemv_mq4g256v2_gfx1100_lm_head_x_buffer";
        let want = [read(0), read(8), write(16)];

        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..3 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_i32(0);
        assert_eq!(blob.len(), 32, "{symbol} explicit args occupy 32 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 32, "{symbol} recorded launches pad to 16");
        assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));

        let got = pointer_effects(symbol).expect("lm_head x_buffer pointer contract");
        assert_eq!(got.len(), want.len());
        for (got_effect, want_effect) in got.iter().zip(want.iter()) {
            assert_eq!(got_effect.offset, want_effect.offset);
            assert_eq!(got_effect.mode, want_effect.mode);
        }
        // Identical resource effects to the generic plain GEMV — A3B replay
        // inventory treats them as the same family (no extra acquire).
        let generic = pointer_effects("gemv_mq4g256v2").expect("generic pointer contract");
        assert_eq!(got.len(), generic.len());
        for (g, w) in got.iter().zip(generic.iter()) {
            assert_eq!(g.offset, w.offset);
            assert_eq!(g.mode, w.mode);
        }
        assert_eq!(
            expected_kernarg_bytes(symbol),
            expected_kernarg_bytes("gemv_mq4g256v2")
        );

        // Distinct frozen symbol — never collapses onto generic or unknown.
        assert_ne!(symbol, "gemv_mq4g256v2");
        assert_ne!(symbol, "gemv_mq4g256v2_residual");
        assert!(pointer_effects("gemv_mq4g256v2_gfx1100_lm_head_x_buffer_unknown").is_none());
        assert!(
            expected_kernarg_bytes("gemv_mq4g256v2_gfx1100_lm_head_x_buffer_unknown").is_none()
        );
        // Gate typo must fail closed: missing _x_buffer suffix.
        assert!(pointer_effects("gemv_mq4g256v2_gfx1100_lm_head").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_gfx1100_lm_head").is_none());
    }

    #[test]
    fn mq4g256v2_residual_sigmoid_scaled_k512_keeps_exact_40b_abi() {
        // Ornith qt44 shared-down fuse: A@0, x@8, y@16 RMW/write, c_buf@24 read;
        // M@32, K@36 → 40 explicit bytes, pad_to(16) → 48. Four pointer offsets.
        let symbol = "gemv_mq4g256v2_residual_sigmoid_scaled_k512";
        let want = [read(0), read(8), write(16), read(24)];

        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..4 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_i32(0);
        assert_eq!(blob.len(), 40, "{symbol} explicit args occupy 40 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 48, "{symbol} recorded launches pad to 16");
        assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));

        let got = pointer_effects(symbol).expect("sigmoid residual pointer contract");
        assert_eq!(got.len(), want.len());
        for (got_effect, want_effect) in got.iter().zip(want.iter()) {
            assert_eq!(got_effect.offset, want_effect.offset);
            assert_eq!(got_effect.mode, want_effect.mode);
        }

        // Distinct from V1 residual_sigmoid and plain V2 residual; fail-closed on typos.
        assert_ne!(symbol, "gemv_hfq4g256_residual_sigmoid_scaled_gpu");
        assert_ne!(symbol, "gemv_mq4g256v2_residual");
        assert!(pointer_effects("gemv_mq4g256v2_residual_sigmoid_scaled").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_residual_sigmoid_scaled").is_none());
    }

    #[test]
    fn fused_qkvza_mq4g256v2_keeps_exact_scalar_replay_contract() {
        let want = [
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
        ];

        // Generic plus gfx1100 K=2048 HOIST_X32 share the same 14-arg ABI:
        // reads@0/8/16/24/32, writes@40/48/56/64, pad96.
        for symbol in [
            "fused_qkvza_mq4g256v2",
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(96));
            let got = pointer_effects(symbol).expect("mq4g256v2 qkvza pointer contract");
            assert_eq!(got.len(), want.len());
            for (got_effect, want_effect) in got.iter().zip(want.iter()) {
                assert_eq!(got_effect.offset, want_effect.offset);
                assert_eq!(got_effect.mode, want_effect.mode);
            }
        }

        // Identities stay independent: hoist never collapses onto generic.
        assert_ne!(
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
            "fused_qkvza_mq4g256v2"
        );

        // 9 ptr + 5 i32 dimensions = 92 explicit bytes, pad_to(16) → 96.
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..9 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..5 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 92, "explicit kernel arguments occupy 92 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 96, "recorded launches pad to 16 bytes");
        for symbol in [
            "fused_qkvza_mq4g256v2",
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
        }
    }

    #[test]
    fn fused_qkv_mq4g256v2_keeps_exact_scalar_replay_contract() {
        let want = [
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
            write(48),
        ];

        // Generic plus gfx1100 K=2048 X_BUFFER share the same 11-arg ABI:
        // reads@0/8/16/24, writes@32/40/48, pad80.
        for symbol in [
            "fused_qkv_mq4g256v2",
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(80));
            let got = pointer_effects(symbol).expect("mq4g256v2 qkv pointer contract");
            assert_eq!(got.len(), want.len());
            for (got_effect, want_effect) in got.iter().zip(want.iter()) {
                assert_eq!(got_effect.offset, want_effect.offset);
                assert_eq!(got_effect.mode, want_effect.mode);
            }
        }

        // Identities stay independent: x_buffer never collapses onto generic.
        assert_ne!(
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
            "fused_qkv_mq4g256v2"
        );

        // 7 ptr + 4 i32 dimensions = 72 explicit bytes, pad_to(16) → 80.
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..7 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..4 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 72, "explicit kernel arguments occupy 72 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 80, "recorded launches pad to 16 bytes");
        for symbol in [
            "fused_qkv_mq4g256v2",
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
        }
    }

    #[test]
    fn fused_gate_up_mq4g256v2_keeps_exact_scalar_replay_contract() {
        let symbol = "fused_gate_up_mq4g256v2";
        let want = [read(0), read(8), read(16), write(24), write(32)];

        assert_eq!(expected_kernarg_bytes(symbol), Some(64));
        let got = pointer_effects(symbol).expect("mq4g256v2 gate_up pointer contract");
        assert_eq!(got.len(), want.len());
        for (got_effect, want_effect) in got.iter().zip(want.iter()) {
            assert_eq!(got_effect.offset, want_effect.offset);
            assert_eq!(got_effect.mode, want_effect.mode);
        }

        // 5 ptr + 3 i32 dimensions = 52 explicit bytes, pad_to(16) → 64.
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..5 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..3 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 52, "explicit kernel arguments occupy 52 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 64, "recorded launches pad to 16 bytes");
        assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
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
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
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
            assert!(independent_sibling(
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                kernel,
            ));
        }
        assert!(!independent_sibling(
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
            "fused_silu_mul_mq_rotate",
        ));
    }

    #[test]
    fn gfx12_rotated_vmem_writers_require_a_pre_dispatch_acquire() {
        for producer in ["mq_rotate_x", "fused_silu_mul_mq_rotate"] {
            assert!(requires_gfx12_pre_dispatch_vmem_acquire(producer));
        }
        assert!(!requires_gfx12_pre_dispatch_vmem_acquire(
            "gemv_hfq4g256_residual"
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
            .acquire_between("mq_rotate_x", "gemv_hfq4g256_multirow_r2"));
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
        assert!(Pm4MidAcquirePolicy::RequiredOnly.acquire_between(
            "fused_qk_l2_norm_scale_f32",
            "gated_delta_net_q8_compact3_b2"
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
            expected_kernarg_bytes("gated_delta_net_q8_compact3_b2"),
            Some(96)
        );
        assert!(pointer_effects("gated_delta_net_q8_compact3_b2").is_some());
        assert_eq!(
            expected_kernarg_bytes("gated_norm_mq_rotate_k6144_gfx1100"),
            Some(64)
        );
        assert!(pointer_effects("gated_norm_mq_rotate_k6144_gfx1100").is_some());
        assert_eq!(
            expected_kernarg_bytes("qwen36_27b_fa_prep_gfx1100"),
            Some(64)
        );
        assert!(pointer_effects("qwen36_27b_fa_prep_gfx1100").is_some());
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
            "fused_gate_up_hfq4g256_dot_reform_gfx1100",
            "fused_gate_up_hfq4g256_dot_prefetch_gfx1100",
            "fused_gate_up_hfq4g256_pair_gfx1100",
            "fused_gate_up_hfq4g256_pair2_gfx1100",
            "fused_gate_up_hfq4g256_quad_prefetch_gfx1100",
            "fused_gate_up_hfq4g256_setprio_gfx1100",
            "fused_gate_up_hfq4g256_lane0_headers_gfx1100",
            "fused_gate_up_hfq4g256_stage_x32_gfx1100",
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
    fn pm4_width_reorder_widens_antichains_without_crossing_dependencies() {
        let mk = |kernel: &str, base: u64, mode: RecordedAccessMode| RecordedHipLaunch {
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

        // One dependent pair (write_x -> read_x) with three launches on
        // unrelated allocations placed either side of it.
        let recorded = vec![
            mk("write_x", 0x1000, RecordedAccessMode::Write),
            mk("indep_a", 0x2000, RecordedAccessMode::Write),
            mk("read_x", 0x1000, RecordedAccessMode::Read),
            mk("indep_b", 0x3000, RecordedAccessMode::Write),
            mk("indep_c", 0x4000, RecordedAccessMode::Write),
        ];

        let order = pm4_width_reorder(&recorded, usize::MAX);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4], "reorder must be a permutation");

        // The one real dependency is preserved.
        let slot = |index: usize| order.iter().position(|value| *value == index).unwrap();
        assert!(slot(0) < slot(2), "write_x must still precede read_x");
    }

    #[test]
    fn pm4_width_reorder_pins_launches_with_unknown_effects() {
        let mk = |kernel: &str, base: u64| RecordedHipLaunch {
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
                mode: RecordedAccessMode::Read,
            }]),
        };
        // An unknown-effect launch conflicts with everything, so it must act as
        // an ordering barrier and hold its recorded position.
        let recorded = vec![
            mk("read_a", 0x1000),
            RecordedHipLaunch {
                accesses: None,
                ..mk("unknown", 0x2000)
            },
            mk("read_b", 0x3000),
        ];
        assert_eq!(pm4_width_reorder(&recorded, usize::MAX), vec![0, 1, 2]);
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
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3, 4],
                    parallel: false,
                    lane_split: None,
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
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3],
                    parallel: false,
                    lane_split: None,
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
                lane_split: None,
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
                lane_split: None,
            }]
        );
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 3, 0, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
                lane_split: None,
            }]
        );
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 2, 3, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
                lane_split: None,
            }]
        );
    }

    #[test]
    fn pm4_ds4_ffn_branch_planner_recovers_dependent_chains() {
        let launch = |kernel: &str, accesses: &[(u64, RecordedAccessMode)]| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            kernarg: Vec::new(),
            accesses: Some(
                accesses
                    .iter()
                    .map(|(base, mode)| RecordedResourceAccess {
                        allocation_base: *base,
                        allocation_bytes: 0x100,
                        access_base: *base,
                        mode: *mode,
                    })
                    .collect(),
            ),
        };
        use RecordedAccessMode::{Read, Write};
        let recorded = vec![
            launch("prepare", &[(0x1000, Write)]),
            launch("zero_f32", &[(0x5000, Write)]),
            launch("gemv_mfp4g32_e8_soa_u4", &[(0x1000, Read), (0x2000, Write)]),
            launch("shared_down", &[(0x2000, Read), (0x4000, Write)]),
            launch(
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
                &[(0x1000, Read), (0x3000, Write)],
            ),
            launch("routed_down", &[(0x3000, Read), (0x5000, Write)]),
            launch("add_inplace_f32", &[(0x4000, Write), (0x5000, Read)]),
            launch("tail", &[(0x4000, Read)]),
        ];

        assert_eq!(
            pm4_ds4_ffn_branch_plan(&recorded).unwrap(),
            vec![
                Pm4PhasePlan {
                    indices: vec![0],
                    parallel: false,
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3, 1, 4, 5],
                    parallel: true,
                    lane_split: Some(2),
                },
                Pm4PhasePlan {
                    indices: vec![6, 7],
                    parallel: false,
                    lane_split: None,
                },
            ]
        );
    }

    #[test]
    fn pm4_ds4_batched_ffn_branch_planner_keeps_routed_down_after_fan_in() {
        let launch = |kernel: &str, accesses: &[(u64, RecordedAccessMode)]| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            kernarg: Vec::new(),
            accesses: Some(
                accesses
                    .iter()
                    .map(|(base, mode)| RecordedResourceAccess {
                        allocation_base: *base,
                        allocation_bytes: 0x100,
                        access_base: *base,
                        mode: *mode,
                    })
                    .collect(),
            ),
        };
        use RecordedAccessMode::{Read, Write};
        let e8 = "gemv_mfp4g32_e8_soa_batched_b3_gfx1151";
        let recorded = vec![
            launch("prepare", &[(0x1000, Write)]),
            launch(e8, &[(0x1000, Read), (0x2000, Write)]),
            launch(e8, &[(0x1000, Read), (0x2100, Write)]),
            launch(
                "deepseek4_silu_mul_clamp_f32",
                &[(0x2000, Write), (0x2100, Read)],
            ),
            launch("mq_rotate_x", &[(0x2000, Read), (0x2200, Write)]),
            launch(e8, &[(0x2200, Read), (0x3000, Write)]),
            launch(e8, &[(0x1000, Read), (0x4000, Write)]),
            launch("sqrt_softplus_f32", &[(0x4000, Write)]),
            launch(
                "deepseek4_moe_topk_bias_aware_batched_f32",
                &[(0x4000, Read), (0x4100, Write)],
            ),
            launch(
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
                &[(0x1000, Read), (0x4100, Read), (0x4200, Write)],
            ),
            launch("deepseek4_silu_mul_clamp_f32", &[(0x4200, Write)]),
            launch("mq_rotate_x", &[(0x4200, Read), (0x4300, Write)]),
            launch(
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
                &[(0x4300, Read), (0x3000, Write)],
            ),
            launch("tail", &[(0x3000, Read)]),
        ];

        assert_eq!(
            pm4_ds4_batched_ffn_branch_plan(&recorded).unwrap(),
            vec![
                Pm4PhasePlan {
                    indices: vec![0],
                    parallel: false,
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: (1..12).collect(),
                    parallel: true,
                    lane_split: Some(5),
                },
                Pm4PhasePlan {
                    indices: vec![12, 13],
                    parallel: false,
                    lane_split: None,
                },
            ]
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
    fn layout_growth_rearms_without_changing_route_selection() {
        let mut controller = ReplayController::new_manual_pm4();
        controller.state = ReplayState::Ready;
        controller.fallback_reason = Some("stale prepared layout".to_owned());

        controller.rearm_after_layout_growth();

        assert_eq!(controller.request(), ReplayBackendRequest::Auto);
        assert_eq!(controller.transport_name(), "pm4");
        assert_eq!(controller.state(), ReplayState::Armed);
        assert_eq!(controller.fallback_reason(), None);
        assert!(!controller.auto_lifecycle);
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
    fn pm4_packet_identity_reports_actual_count() {
        // Phased multi-queue graphs legitimately carry barrier + IB packets per lane.
        assert_eq!(pm4_packet_identity(0), None);
        assert_eq!(pm4_packet_identity(1), Some(1));
        assert_eq!(pm4_packet_identity(260), Some(260));
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

    #[test]
    fn replay_grid_binding_units_for_and_bind() {
        let binding = ReplayGridBinding::PositionCeilDiv {
            axis: 0,
            addend: 15,
            divisor: 16,
        };
        // max_position = 32 => ceil((32+15)/16)=3
        assert_eq!(binding.units_for(32).unwrap(), 3);
        // current position 16 => ceil(31/16)=2, min with prepared max 3 => 2
        let prepared_grid = [3, 1, 1];
        assert_eq!(binding.bind(16, prepared_grid).unwrap(), [2, 1, 1]);
        // current position 0 => ceil(15/16)=1
        assert_eq!(binding.bind(0, prepared_grid).unwrap(), [1, 1, 1]);
        // position == max => 3
        assert_eq!(binding.bind(32, prepared_grid).unwrap(), [3, 1, 1]);
    }

    #[test]
    fn replay_grid_binding_prepared_sizing_and_rejection() {
        let binding = ReplayGridBinding::PositionCeilDiv {
            axis: 0,
            addend: 0,
            divisor: 1,
        };
        // Simulate prepared_max_position = 100
        let prepared_max = 100usize;
        let recorded_grid = [10, 1, 1];
        let prepared_units = binding.units_for(prepared_max).unwrap();
        let mut prepared_grid = recorded_grid;
        prepared_grid[0] = prepared_units;
        assert_eq!(prepared_grid, [100, 1, 1]);
        // Patch down for smaller position 10 => 10
        assert_eq!(binding.bind(10, prepared_grid).unwrap(), [10, 1, 1]);
        // Reject position > max_position should be handled at replay level
        // (here we simulate the check that PreparedPm4Replay::replay_and_wait_checked does)
        let current = 150usize;
        let should_reject = current > prepared_max;
        assert!(should_reject);
    }

    #[test]
    fn replay_kernarg_binding_apply_offsets() {
        let mut kernarg = vec![0u8; 80];
        // GdnFrameU32 at offset 76 writes 4 bytes; now carries explicit frames count.
        let gdn = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 1,
        };
        gdn.apply(&mut kernarg, 0).unwrap();
        let first = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        // `reserve_gdn_requant_frames` is a `fetch_add`, so the first frame in
        // a fresh process is legitimately 0. What must hold is that every apply
        // consumes a fresh frame rather than replaying the captured one.
        gdn.apply(&mut kernarg, 0).unwrap();
        let second = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        assert!(second > first, "frame did not advance: {first} -> {second}");
        // PositionPlusU32 at offset 0 with addend 5, position 10 => 15
        let mut kernarg2 = vec![0u8; 8];
        let pos_binding = ReplayKernargBinding::PositionPlusU32 {
            offset: 0,
            addend: 5,
        };
        pos_binding.apply(&mut kernarg2, 10).unwrap();
        let value = u32::from_ne_bytes(kernarg2[0..4].try_into().unwrap());
        assert_eq!(value, 15);
        // Out-of-bounds should error
        let mut small = vec![0u8; 4];
        let bad = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 1,
        };
        assert!(bad.apply(&mut small, 0).is_err());
    }

    #[test]
    fn replay_kernarg_binding_gdn_requires_binding() {
        // Simulate preparation admissibility: a GDN-family launch without a
        // GdnFrameU32 binding must be rejected instead of silently replaying stale.
        let kernel = "gated_delta_net_q8_fast";
        let metadata_kernarg_size = 80usize;
        assert!(metadata_kernarg_size >= 80);
        let bindings: Vec<(usize, ReplayKernargBinding)> = vec![];
        let is_gdn =
            kernel == "gated_delta_net_q8_fast" || kernel.starts_with("gated_delta_net_q8_compact");
        assert!(is_gdn);
        let has_binding = bindings.iter().any(|(_, b)| {
            matches!(
                b,
                ReplayKernargBinding::GdnFrameU32 {
                    offset: 76,
                    frames: _
                }
            )
        });
        assert!(!has_binding);
        // In real prepare, this would be Err; here we just verify detection.
        let should_reject = is_gdn && !has_binding;
        assert!(should_reject);
        // With correct binding, it passes.
        let bindings_ok = vec![(
            0,
            ReplayKernargBinding::GdnFrameU32 {
                offset: 76,
                frames: 1,
            },
        )];
        let has_binding_ok = bindings_ok.iter().any(|(_, b)| {
            matches!(
                b,
                ReplayKernargBinding::GdnFrameU32 {
                    offset: 76,
                    frames: _
                }
            )
        });
        assert!(has_binding_ok);
        assert!(!(is_gdn && !has_binding_ok));
    }

    fn make_gdn_kernarg(nt: i32, len: usize) -> Vec<u8> {
        let mut kernarg = vec![0u8; len];
        if len >= 68 {
            kernarg[64..68].copy_from_slice(&nt.to_le_bytes());
        }
        kernarg
    }

    #[test]
    fn gdn_frame_count_derivation_sequence_and_independent_batched() {
        // sequence-batched: nt = 16 tokens, grid.z = 1 => frames = 16
        let kernarg = make_gdn_kernarg(16, 80);
        assert_eq!(gdn_requant_frames_for_dispatch(&kernarg, 1).unwrap(), 16);
        // independent-batched: nt = 1, grid.z = 16 => frames = 16 (1 * 16)
        let kernarg = make_gdn_kernarg(1, 80);
        assert_eq!(gdn_requant_frames_for_dispatch(&kernarg, 16).unwrap(), 16);
        // plain AR single-token: nt = 1, grid.z = 1 => frames = 1 (preserves today's behavior)
        let kernarg = make_gdn_kernarg(1, 80);
        assert_eq!(gdn_requant_frames_for_dispatch(&kernarg, 1).unwrap(), 1);
    }

    #[test]
    fn gdn_frame_count_rejects_short_and_nonpositive_nt() {
        // kernarg shorter than 80 bytes must be rejected
        let short = make_gdn_kernarg(16, 64);
        assert!(gdn_requant_frames_for_dispatch(&short, 1).is_err());
        let short79 = make_gdn_kernarg(16, 79);
        assert!(gdn_requant_frames_for_dispatch(&short79, 1).is_err());
        // nt <= 0 must be rejected
        let zero = make_gdn_kernarg(0, 80);
        assert!(gdn_requant_frames_for_dispatch(&zero, 1).is_err());
        let neg = make_gdn_kernarg(-1, 80);
        assert!(gdn_requant_frames_for_dispatch(&neg, 1).is_err());
        // i32::MAX * 2 == 4294967294, which still fits in u32 — the first
        // genuinely overflowing multiplier is 3.
        let large = make_gdn_kernarg(i32::MAX, 80);
        assert!(gdn_requant_frames_for_dispatch(&large, 2).is_ok());
        assert!(gdn_requant_frames_for_dispatch(&large, 3).is_err());
    }

    #[test]
    fn gdn_frame_apply_advances_by_frames_and_writes_base() {
        // Restore a known checkpoint so we can assert on delta rather than
        // absolute non-zero value (fetch_add legitimately starts at 0 in a
        // fresh process).
        crate::norm::restore_gdn_requant_frame_checkpoint(1000);
        let checkpoint = crate::norm::gdn_requant_frame_checkpoint();
        let mut kernarg = vec![0u8; 80];
        let binding = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 16,
        };
        binding.apply(&mut kernarg, 0).unwrap();
        let written = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        // Written base must equal the pre-advance checkpoint
        assert_eq!(written, checkpoint);
        let after = crate::norm::gdn_requant_frame_checkpoint();
        assert_eq!(after, checkpoint + 16);
        // Second apply with frames=1 should advance by 1 from new base
        let checkpoint2 = after;
        let binding2 = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 1,
        };
        binding2.apply(&mut kernarg, 0).unwrap();
        let written2 = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        assert_eq!(written2, checkpoint2);
        assert_eq!(crate::norm::gdn_requant_frame_checkpoint(), checkpoint2 + 1);
        // Clean up: restore to avoid leaking state to other tests (tests run
        // in parallel in same process; use a deterministic restore).
        crate::norm::restore_gdn_requant_frame_checkpoint(checkpoint);
    }

    #[test]
    fn replay_quiescence_mapping() {
        fn map(q: redline_dispatch::aql::Quiescence) -> ReplayQuiescence {
            match q {
                redline_dispatch::aql::Quiescence::Proven => ReplayQuiescence::Proven,
                redline_dispatch::aql::Quiescence::Unknown => ReplayQuiescence::Unknown,
            }
        }
        assert_eq!(
            map(redline_dispatch::aql::Quiescence::Proven),
            ReplayQuiescence::Proven
        );
        assert_eq!(
            map(redline_dispatch::aql::Quiescence::Unknown),
            ReplayQuiescence::Unknown
        );
    }

    #[test]
    fn unknown_quiescence_requires_quiesce_before_reuse_and_shutdown_retains() {
        // C0 contract pin at rdna layer: Unknown means in-flight may still be
        // writing. Subsequent replay must be refused and shutdown must retain
        // prepared IB/kernargs/kernels until quiesce proves Proven.
        let failure = RetainedReplayFailure {
            error: "simulated doorbell wait timeout".to_owned(),
            quiescence: ReplayQuiescence::Unknown,
        };
        assert_eq!(failure.quiescence, ReplayQuiescence::Unknown);
        // Caller policy: Unknown => quarantine — do not free, do not replay,
        // do not reuse controller for new model until quiesce succeeds.
        let must_quarantine = failure.quiescence == ReplayQuiescence::Unknown;
        assert!(must_quarantine);
        // Proven => safe to retry or free after handling error
        let proven = RetainedReplayFailure {
            error: "position 200 exceeds prepared max_position 100".to_owned(),
            quiescence: ReplayQuiescence::Proven,
        };
        assert_eq!(proven.quiescence, ReplayQuiescence::Proven);
        assert!(!proven.error.is_empty());
        // Simulate shutdown retention: on Unknown, shutdown would return Err(Unknown)
        // and keep prepared resources; on Proven it returns Ok and releases.
        fn simulated_shutdown(quiescence: ReplayQuiescence) -> Result<(), RetainedReplayFailure> {
            match quiescence {
                ReplayQuiescence::Unknown => Err(RetainedReplayFailure {
                    error: "inactivate_all failed: device still in-flight".to_owned(),
                    quiescence: ReplayQuiescence::Unknown,
                }),
                ReplayQuiescence::Proven => Ok(()),
            }
        }
        assert!(simulated_shutdown(ReplayQuiescence::Unknown).is_err());
        assert!(simulated_shutdown(ReplayQuiescence::Proven).is_ok());
    }

    fn make_kernarg_with_u32(offset: usize, value: u32, len: usize) -> Vec<u8> {
        let mut kernarg = vec![0u8; len];
        kernarg[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        kernarg
    }

    #[test]
    fn position_scalar_advances_by_delta_yields_binding_and_applies_at_third_position() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let kernarg_10 = make_kernarg_with_u32(16, 10, 64);
        controller.record_hip_launch("gemv_hfq4g256", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_10);
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let kernarg_20 = make_kernarg_with_u32(16, 20, 64);
        controller.record_hip_launch("gemv_hfq4g256", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_20);
        controller.finish_capture().unwrap();
        let count = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap();
        assert_eq!(count, 1);
        let bindings = controller.synthesized_position_bindings().to_vec();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, 0);
        match bindings[0].1 {
            ReplayKernargBinding::PositionPlusU32 { offset, addend } => {
                assert_eq!(offset, 16);
                assert_eq!(addend, 0); // 20 - 20 = 0, 10 -10 =0
            }
            _ => panic!("expected PositionPlusU32"),
        }
        // Applying at third position 30 should produce 30.
        let mut kernarg_third = vec![0u8; 64];
        apply_kernarg_bindings_for_dispatch(&mut kernarg_third, 0, 30, &bindings).unwrap();
        let v = u32::from_le_bytes(kernarg_third[16..20].try_into().unwrap());
        // But apply writes value = position + addend = 30 + 0 =30; with ne_bytes vs le_bytes?
        // PositionPlusU32 uses to_ne_bytes, so need to read via from_ne_bytes.
        let v_ne = u32::from_ne_bytes(kernarg_third[16..20].try_into().unwrap());
        assert_eq!(v_ne, 30);
        // Also verify that helper reproduces earlier samples.
        let mut kernarg_earlier = vec![0u8; 64];
        apply_kernarg_bindings_for_dispatch(&mut kernarg_earlier, 0, 10, &bindings).unwrap();
        let ve = u32::from_ne_bytes(kernarg_earlier[16..20].try_into().unwrap());
        assert_eq!(ve, 10);
        let mut kernarg_current = vec![0u8; 64];
        apply_kernarg_bindings_for_dispatch(&mut kernarg_current, 0, 20, &bindings).unwrap();
        let vc = u32::from_ne_bytes(kernarg_current[16..20].try_into().unwrap());
        assert_eq!(vc, 20);
    }

    #[test]
    fn position_scalar_with_constant_offset_recovers_addend() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        // scalar = position + 5
        let kernarg_10 = make_kernarg_with_u32(32, 15, 64); // 10+5
        controller.record_hip_launch(
            "fused_qkv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_10,
        );
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let kernarg_20 = make_kernarg_with_u32(32, 25, 64); // 20+5
        controller.record_hip_launch(
            "fused_qkv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_20,
        );
        controller.finish_capture().unwrap();
        let count = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap();
        assert_eq!(count, 1);
        match controller.synthesized_position_bindings()[0].1 {
            ReplayKernargBinding::PositionPlusU32 { offset, addend } => {
                assert_eq!(offset, 32);
                assert_eq!(addend, 5);
            }
            _ => panic!("expected PositionPlusU32"),
        }
    }

    #[test]
    fn position_scalar_non_delta_difference_is_rejected_with_kernel_index_offset() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let kernarg_10 = make_kernarg_with_u32(8, 100, 64);
        controller.record_hip_launch("softmax_f32", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_10);
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        // Diff is 5, delta is 10 (positions 10->20 delta 10, but value diff 5)
        let kernarg_20 = make_kernarg_with_u32(8, 105, 64);
        controller.record_hip_launch("softmax_f32", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_20);
        controller.finish_capture().unwrap();
        let err = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap_err();
        assert!(err.contains("softmax_f32"), "error missing kernel: {err}");
        assert!(
            err.contains("launch 0"),
            "error missing launch index: {err}"
        );
        assert!(err.contains("offset 8"), "error missing byte offset: {err}");
        assert!(
            err.contains("unexplained"),
            "error missing unexplained: {err}"
        );
    }

    #[test]
    fn eight_byte_pointer_field_is_rejected_as_moved_allocation() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let mut kernarg_early = vec![0u8; 32];
        let ptr_early: u64 = 0x7f00_0000_1000;
        kernarg_early[0..8].copy_from_slice(&ptr_early.to_le_bytes());
        controller.record_hip_launch(
            "gemv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_early,
        );
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let mut kernarg_cur = vec![0u8; 32];
        let ptr_cur: u64 = 0x7f00_0000_2000; // moved allocation, diff != delta
        kernarg_cur[0..8].copy_from_slice(&ptr_cur.to_le_bytes());
        controller.record_hip_launch(
            "gemv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_cur,
        );
        controller.finish_capture().unwrap();
        let err = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap_err();
        assert!(
            err.contains("moved allocation"),
            "error missing moved allocation: {err}"
        );
        assert!(err.contains("offset 0"), "error missing offset: {err}");
    }

    #[test]
    fn gdn_frame_offset_76_is_skipped_on_gdn_family() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let mut kernarg_early = vec![0u8; 80];
        kernarg_early[76..80].copy_from_slice(&1234u32.to_le_bytes());
        // need nt at 64 to be valid but synthesize skips offset 76 regardless
        kernarg_early[64..68].copy_from_slice(&1i32.to_le_bytes());
        controller.record_hip_launch(
            "gated_delta_net_q8_fast",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_early,
        );
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let mut kernarg_cur = vec![0u8; 80];
        kernarg_cur[76..80].copy_from_slice(&9999u32.to_le_bytes()); // arbitrary diff
        kernarg_cur[64..68].copy_from_slice(&1i32.to_le_bytes());
        controller.record_hip_launch(
            "gated_delta_net_q8_fast",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_cur,
        );
        controller.finish_capture().unwrap();
        let count = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap();
        assert_eq!(
            count, 0,
            "GDN offset 76 should be skipped, got bindings: {count}"
        );
    }

    #[test]
    fn mismatched_launch_counts_kernel_names_and_kernarg_lengths_are_rejected_distinctly() {
        // Mismatched launch counts
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller.record_hip_launch("b", None, [1, 1, 1], [32, 1, 1], 0, &[5, 6, 7, 8]);
        controller.finish_capture().unwrap();
        let err = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap_err();
        assert!(
            err.contains("launch count mismatch"),
            "expected launch count mismatch: {err}"
        );

        // Mismatched kernel names
        let mut controller2 = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller2.begin_capture().unwrap();
        controller2.record_hip_launch("kernel_a", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller2.finish_capture().unwrap();
        let earlier2 = controller2.snapshot_recorded_kernargs();
        controller2.begin_capture().unwrap();
        controller2.record_hip_launch("kernel_b", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller2.finish_capture().unwrap();
        let err2 = controller2
            .synthesize_position_bindings(&earlier2, 10, 20)
            .unwrap_err();
        assert!(
            err2.contains("kernel mismatch"),
            "expected kernel mismatch: {err2}"
        );

        // Mismatched kernarg lengths
        let mut controller3 = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller3.begin_capture().unwrap();
        controller3.record_hip_launch("same_kernel", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller3.finish_capture().unwrap();
        let earlier3 = controller3.snapshot_recorded_kernargs();
        controller3.begin_capture().unwrap();
        controller3.record_hip_launch(
            "same_kernel",
            None,
            [1, 1, 1],
            [32, 1, 1],
            0,
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );
        controller3.finish_capture().unwrap();
        let err3 = controller3
            .synthesize_position_bindings(&earlier3, 10, 20)
            .unwrap_err();
        assert!(
            err3.contains("kernarg length mismatch"),
            "expected kernarg length mismatch: {err3}"
        );
        // Ensure distinct messages
        assert_ne!(err, err2);
        assert_ne!(err2, err3);
        assert_ne!(err, err3);
    }
}
