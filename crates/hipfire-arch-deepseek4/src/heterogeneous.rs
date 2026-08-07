// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transactional two-device ownership for DeepSeek V4 MQ2R.
//!
//! This module is intentionally architecture-local. It does not use the
//! process-wide mixed-architecture escape hatch and cannot affect Qwen's
//! single-device or homogeneous multi-GPU loaders.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use hip_bridge::{DeviceBuffer, Event, HipRuntime, Stream};
use hipfire_config::{Deepseek4ComputePlacement, DeviceSelector};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::arch::{DeepseekV4HeterogeneousFault, DeepseekV4HeterogeneousProjection};
use crate::deepseek4::{
    DeepseekV4Config, DeepseekV4HeterogeneousWeights, DeepseekV4OwnershipAudit, DeepseekV4State,
};
use crate::forward::PrefillBatchScratch;
use crate::DeepseekV4;

pub const MQ2R_0731_SHA256: &str =
    "cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce";
pub const DEFAULT_SAFETY_MARGIN_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const HARMONIC_EXECUTION_QUARANTINE: &str =
    "deepseek4 harmonic execution is quarantined: the former split-owner route used reciprocal indefinite cross-device HIP waits; use single placement until H2 fault-containment certification removes this guard";

/// Fail closed before artifact hashing, HIP discovery, allocation, or queue
/// creation. H0 deliberately preserves the accepted ownership and kernel work
/// while making the unsafe product transport unreachable.
pub fn ensure_harmonic_execution_admitted() -> Result<(), String> {
    Err(HARMONIC_EXECUTION_QUARANTINE.into())
}

/// Content identity receipt for repeated transactional tests or reloads in one
/// process. The expensive 82 GiB SHA scan happens once; every reuse proves the
/// canonical path, length, and modification time are unchanged before opening
/// the artifact again.
#[derive(Debug, Clone)]
pub struct DeepseekV4VerifiedArtifact {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
    pub sha256: String,
}

impl DeepseekV4VerifiedArtifact {
    pub fn verify(path: &Path) -> Result<Self, String> {
        let path = path
            .canonicalize()
            .map_err(|error| format!("deepseek4 canonicalize {}: {error}", path.display()))?;
        let metadata = path
            .metadata()
            .map_err(|error| format!("deepseek4 metadata {}: {error}", path.display()))?;
        let modified = metadata
            .modified()
            .map_err(|error| format!("deepseek4 mtime {}: {error}", path.display()))?;
        let sha256 = crate::parent::manifest::sha256_file(&path)?;
        if sha256 != MQ2R_0731_SHA256 {
            return Err(format!(
                "deepseek4 heterogeneous artifact SHA mismatch: got {sha256}, expected {MQ2R_0731_SHA256}"
            ));
        }
        Ok(Self {
            path,
            len: metadata.len(),
            modified,
            sha256,
        })
    }

    pub(crate) fn validate(&self, path: &Path) -> Result<String, String> {
        let canonical = path
            .canonicalize()
            .map_err(|error| format!("deepseek4 canonicalize {}: {error}", path.display()))?;
        if canonical != self.path {
            return Err(format!(
                "deepseek4 verified artifact path changed: {} != {}",
                canonical.display(),
                self.path.display()
            ));
        }
        let metadata = canonical
            .metadata()
            .map_err(|error| format!("deepseek4 metadata {}: {error}", canonical.display()))?;
        let modified = metadata
            .modified()
            .map_err(|error| format!("deepseek4 mtime {}: {error}", canonical.display()))?;
        if metadata.len() != self.len || modified != self.modified {
            return Err("deepseek4 verified artifact changed after identity scan".into());
        }
        Ok(self.sha256.clone())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV4HeterogeneousLoadPlan {
    pub placement: Deepseek4ComputePlacement,
    pub prefill_max_batch: usize,
    pub safety_margin_bytes: usize,
}

impl Default for DeepseekV4HeterogeneousLoadPlan {
    fn default() -> Self {
        Self {
            placement: Deepseek4ComputePlacement::DenseExpertSplit {
                dense: DeviceSelector::ExactArch("gfx1100".into()),
                experts: DeviceSelector::ExactArch("gfx1151".into()),
            },
            prefill_max_batch: 1024,
            safety_margin_bytes: DEFAULT_SAFETY_MARGIN_BYTES,
        }
    }
}

impl DeepseekV4HeterogeneousLoadPlan {
    pub(crate) fn resolve_device_ids(&self) -> Result<(i32, i32), String> {
        let Deepseek4ComputePlacement::DenseExpertSplit { dense, experts } = &self.placement else {
            return Err(
                "deepseek4 heterogeneous load requires dense-expert-split placement".into(),
            );
        };
        let hip = HipRuntime::load()
            .map_err(|error| format!("deepseek4 heterogeneous HIP discovery: {error}"))?;
        let dense_id = resolve_device_selector(&hip, dense)?;
        let routed_id = resolve_device_selector(&hip, experts)?;
        if dense_id == routed_id {
            return Err(
                "deepseek4 heterogeneous placement resolves both roles to one device".into(),
            );
        }
        Ok((dense_id, routed_id))
    }
}

fn resolve_device_selector(hip: &HipRuntime, selector: &DeviceSelector) -> Result<i32, String> {
    match selector {
        DeviceSelector::ExactArch(expected) => {
            let count = hip
                .device_count()
                .map_err(|error| format!("deepseek4 heterogeneous device count: {error}"))?;
            let mut matches = Vec::new();
            for device_id in 0..count {
                let arch = hip.get_arch(device_id).map_err(|error| {
                    format!("deepseek4 heterogeneous device {device_id} architecture: {error}")
                })?;
                if arch.eq_ignore_ascii_case(expected) {
                    matches.push(device_id);
                }
            }
            match matches.as_slice() {
                [device_id] => {
                    // Architecture names are the portable role selector. Pin
                    // the unique match to HIP's current physical identity for
                    // this process so hotplug/reboot ordinal order can never
                    // change which device owns the role after discovery.
                    let pci_bus_id = hip.device_pci_bus_id(*device_id).map_err(|error| {
                        format!(
                            "deepseek4 heterogeneous selector {selector} PCI identity: {error}"
                        )
                    })?;
                    let pinned = hip.device_by_pci_bus_id(&pci_bus_id).map_err(|error| {
                        format!(
                            "deepseek4 heterogeneous selector {selector} pin {pci_bus_id}: {error}"
                        )
                    })?;
                    if pinned != *device_id {
                        return Err(format!(
                            "deepseek4 heterogeneous selector {selector} changed ordinal during PCI pin: discovered {device_id}, resolved {pinned} at {pci_bus_id}"
                        ));
                    }
                    Ok(pinned)
                }
                [] => Err(format!(
                    "deepseek4 heterogeneous selector {selector} matched no visible device"
                )),
                _ => Err(format!(
                    "deepseek4 heterogeneous selector {selector} matched {} visible devices; use PCI BDF or UUID",
                    matches.len()
                )),
            }
        }
        DeviceSelector::PciBdf(pci_bus_id) => hip
            .device_by_pci_bus_id(pci_bus_id)
            .map_err(|error| format!("deepseek4 heterogeneous selector {selector}: {error}")),
        DeviceSelector::Uuid(_) => Err(format!(
            "deepseek4 heterogeneous selector {selector} is typed but this HIP discovery layer cannot resolve UUIDs yet"
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV4HeterogeneousLoadReport {
    pub model_sha256: String,
    pub projection: DeepseekV4HeterogeneousProjection,
    pub dense_state_scratch_projected_bytes: usize,
    pub ownership: DeepseekV4OwnershipAudit,
    pub dense_free_before: usize,
    pub dense_free_after: usize,
    pub routed_free_before: usize,
    pub routed_free_after: usize,
    pub dense_actual_bytes: usize,
    pub routed_actual_bytes: usize,
    pub dense_state_scratch_pool_bytes: usize,
}

/// Load-scoped owner for the two devices and every successfully constructed
/// resource. Any `?` after device creation reaches this destructor, including
/// pointer-audit, meminfo, bind, state, scratch, and injected failures.
struct DeepseekV4HeterogeneousStaging {
    dense_gpu: Option<Gpu>,
    routed_gpu: Option<Gpu>,
    weights: Option<DeepseekV4HeterogeneousWeights>,
    state: Option<DeepseekV4State>,
    prefill: Option<PrefillBatchScratch>,
}

impl DeepseekV4HeterogeneousStaging {
    fn new(dense_gpu: Gpu, routed_gpu: Gpu) -> Self {
        Self {
            dense_gpu: Some(dense_gpu),
            routed_gpu: Some(routed_gpu),
            weights: None,
            state: None,
            prefill: None,
        }
    }

    fn devices_mut(&mut self) -> (&mut Gpu, &mut Gpu) {
        (
            self.dense_gpu
                .as_mut()
                .expect("dense staging device missing"),
            self.routed_gpu
                .as_mut()
                .expect("routed staging device missing"),
        )
    }

    fn audit_weights(&mut self) -> Result<DeepseekV4OwnershipAudit, String> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| "deepseek4: ownership audit before weight publication".to_string())?;
        let dense_gpu = self
            .dense_gpu
            .as_mut()
            .ok_or_else(|| "deepseek4: dense staging device missing".to_string())?;
        let routed_gpu = self
            .routed_gpu
            .as_mut()
            .ok_or_else(|| "deepseek4: routed staging device missing".to_string())?;
        weights.audit_owners(dense_gpu, routed_gpu)
    }

    fn publish(
        mut self,
        config: DeepseekV4Config,
        report: DeepseekV4HeterogeneousLoadReport,
    ) -> DeepseekV4HeterogeneousModel {
        DeepseekV4HeterogeneousModel {
            dense_gpu: self.dense_gpu.take().expect("dense staging device missing"),
            routed_gpu: self
                .routed_gpu
                .take()
                .expect("routed staging device missing"),
            config,
            weights: self.weights.take(),
            state: self.state.take(),
            prefill: self.prefill.take(),
            execution: None,
            report,
        }
    }

    fn release(&mut self) {
        let (Some(dense_gpu), Some(routed_gpu)) =
            (self.dense_gpu.as_mut(), self.routed_gpu.as_mut())
        else {
            return;
        };
        if let Some(prefill) = self.prefill.take() {
            prefill.free_gpu(dense_gpu);
        }
        if let Some(state) = self.state.take() {
            state.free_gpu(dense_gpu);
        }
        if let Some(weights) = self.weights.take() {
            weights.free_gpu(dense_gpu, routed_gpu);
        }
        dense_gpu.invalidate_weight_caches();
        dense_gpu.invalidate_graph_state();
        routed_gpu.invalidate_weight_caches();
        routed_gpu.invalidate_graph_state();
        dense_gpu.drain_pool();
        routed_gpu.drain_pool();
    }
}

impl Drop for DeepseekV4HeterogeneousStaging {
    fn drop(&mut self) {
        self.release();
    }
}

/// Fully owned G2 transaction. Destruction always frees scratch, state, then
/// each weight class on its exact owner and drains both independent pools.
pub struct DeepseekV4HeterogeneousModel {
    pub dense_gpu: Gpu,
    pub routed_gpu: Gpu,
    pub config: DeepseekV4Config,
    pub weights: Option<DeepseekV4HeterogeneousWeights>,
    pub state: Option<DeepseekV4State>,
    pub prefill: Option<PrefillBatchScratch>,
    execution: Option<DeepseekV4HeterogeneousExecution>,
    pub report: DeepseekV4HeterogeneousLoadReport,
}

/// Persistent direct-HIP execution resources for the split-owner route. Only
/// routed-expert scratch lives on gfx1151; canonical model state remains on
/// gfx1100. Cross-device epochs use the system-visible signal-memory contract
/// certified by G0, so the layer loop never synchronizes on the host.
pub(crate) struct DeepseekV4HeterogeneousExecution {
    /// Exact-gfx1100 side queue for the independent KV/compressor half of the
    /// attention projection DAG. The primary dense queue keeps Q-LoRA; one
    /// reusable event pair brackets the fork/join on every layer.
    pub(crate) dense_attn_stream: Option<Stream>,
    pub(crate) dense_attn_fork_event: Option<Event>,
    pub(crate) dense_attn_join_event: Option<Event>,
    pub(crate) routed_x_rot: Option<GpuTensor>,
    pub(crate) routed_topk_indices: Option<GpuTensor>,
    pub(crate) routed_topk_weights: Option<GpuTensor>,
    pub(crate) routed_gate_batch: Option<GpuTensor>,
    pub(crate) routed_up_batch: Option<GpuTensor>,
    pub(crate) routed_rot_batch: Option<GpuTensor>,
    pub(crate) routed_down_expanded: Option<GpuTensor>,
    pub(crate) routed_partial: Option<GpuTensor>,
    pub(crate) signal_to_dense: Option<DeviceBuffer>,
    pub(crate) signal_to_routed: Option<DeviceBuffer>,
    epoch: u32,
}

impl DeepseekV4HeterogeneousExecution {
    fn empty() -> Self {
        Self {
            dense_attn_stream: None,
            dense_attn_fork_event: None,
            dense_attn_join_event: None,
            routed_x_rot: None,
            routed_topk_indices: None,
            routed_topk_weights: None,
            routed_gate_batch: None,
            routed_up_batch: None,
            routed_rot_batch: None,
            routed_down_expanded: None,
            routed_partial: None,
            signal_to_dense: None,
            signal_to_routed: None,
            epoch: 0,
        }
    }

    fn new(
        dense_gpu: &mut Gpu,
        routed_gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
    ) -> Result<Self, String> {
        if dense_gpu.active_stream.is_some() || routed_gpu.active_stream.is_some() {
            return Err(
                "deepseek4 heterogeneous direct-HIP route requires unclaimed primary streams"
                    .into(),
            );
        }
        let mut execution = Self::empty();
        let result =
            (|| {
                dense_gpu
                    .bind_thread()
                    .map_err(|error| format!("heterogeneous bind dense execution: {error}"))?;
                dense_gpu
                    .hip
                    .enable_peer_access(routed_gpu.device_id)
                    .map_err(|error| format!("heterogeneous dense peer access: {error}"))?;
                dense_gpu.active_stream = Some(
                    dense_gpu
                        .hip
                        .stream_create()
                        .map_err(|error| format!("heterogeneous dense stream: {error}"))?,
                );
                execution.dense_attn_stream =
                    Some(dense_gpu.hip.stream_create().map_err(|error| {
                        format!("heterogeneous dense attention stream: {error}")
                    })?);
                execution.dense_attn_fork_event = Some(
                    dense_gpu
                        .hip
                        .event_create()
                        .map_err(|error| format!("heterogeneous dense attention fork: {error}"))?,
                );
                execution.dense_attn_join_event = Some(
                    dense_gpu
                        .hip
                        .event_create()
                        .map_err(|error| format!("heterogeneous dense attention join: {error}"))?,
                );
                let signal_to_dense = dense_gpu
                    .hip
                    .malloc_signal(std::mem::size_of::<u64>())
                    .map_err(|error| format!("heterogeneous dense signal: {error}"))?;
                dense_gpu
                    .hip
                    .memset(&signal_to_dense, 0, signal_to_dense.size())
                    .map_err(|error| format!("heterogeneous zero dense signal: {error}"))?;
                execution.signal_to_dense = Some(signal_to_dense);

                routed_gpu
                    .bind_thread()
                    .map_err(|error| format!("heterogeneous bind routed execution: {error}"))?;
                routed_gpu
                    .hip
                    .enable_peer_access(dense_gpu.device_id)
                    .map_err(|error| format!("heterogeneous routed peer access: {error}"))?;
                routed_gpu.active_stream = Some(
                    routed_gpu
                        .hip
                        .stream_create()
                        .map_err(|error| format!("heterogeneous routed stream: {error}"))?,
                );
                let signal_to_routed = routed_gpu
                    .hip
                    .malloc_signal(std::mem::size_of::<u64>())
                    .map_err(|error| format!("heterogeneous routed signal: {error}"))?;
                routed_gpu
                    .hip
                    .memset(&signal_to_routed, 0, signal_to_routed.size())
                    .map_err(|error| format!("heterogeneous zero routed signal: {error}"))?;
                execution.signal_to_routed = Some(signal_to_routed);

                let k = cfg.num_experts_per_tok;
                let im = cfg.moe_intermediate_size;
                let hidden = cfg.hidden_size;
                execution.routed_x_rot = Some(
                    routed_gpu
                        .alloc_tensor(&[hidden], DType::F32)
                        .map_err(|error| format!("heterogeneous routed x_rot: {error}"))?,
                );
                execution.routed_topk_indices = Some(
                    routed_gpu
                        .alloc_tensor(&[k], DType::F32)
                        .map_err(|error| format!("heterogeneous routed topk indices: {error}"))?,
                );
                execution.routed_topk_weights = Some(
                    routed_gpu
                        .alloc_tensor(&[k], DType::F32)
                        .map_err(|error| format!("heterogeneous routed topk weights: {error}"))?,
                );
                execution.routed_gate_batch = Some(
                    routed_gpu
                        .alloc_tensor(&[k, im], DType::F32)
                        .map_err(|error| format!("heterogeneous routed gate batch: {error}"))?,
                );
                execution.routed_up_batch = Some(
                    routed_gpu
                        .alloc_tensor(&[k, im], DType::F32)
                        .map_err(|error| format!("heterogeneous routed up batch: {error}"))?,
                );
                execution.routed_rot_batch = Some(
                    routed_gpu
                        .alloc_tensor(&[k, im], DType::F32)
                        .map_err(|error| format!("heterogeneous routed rot batch: {error}"))?,
                );
                execution.routed_down_expanded = Some(
                    routed_gpu
                        .alloc_tensor(&[k, hidden], DType::F32)
                        .map_err(|error| format!("heterogeneous routed down expanded: {error}"))?,
                );
                execution.routed_partial = Some(
                    routed_gpu
                        .alloc_tensor(&[hidden], DType::F32)
                        .map_err(|error| format!("heterogeneous routed partial: {error}"))?,
                );
                Ok(())
            })();
        if let Err(error) = result {
            execution.release(dense_gpu, routed_gpu);
            return Err(error);
        }
        Ok(execution)
    }

    pub(crate) fn next_epoch(&mut self) -> Result<u32, String> {
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| "deepseek4 heterogeneous signal epoch exhausted".to_string())?;
        Ok(self.epoch)
    }

    fn release(&mut self, dense_gpu: &mut Gpu, routed_gpu: &mut Gpu) {
        fn free_opt(gpu: &mut Gpu, tensor: &mut Option<GpuTensor>) {
            if let Some(tensor) = tensor.take() {
                let _ = gpu.free_tensor(tensor);
            }
        }

        if let Some(stream) = dense_gpu.active_stream.as_ref() {
            dense_gpu.bind_thread_or_warn();
            let _ = dense_gpu.hip.stream_synchronize(stream);
        }
        if let Some(stream) = self.dense_attn_stream.as_ref() {
            dense_gpu.bind_thread_or_warn();
            let _ = dense_gpu.hip.stream_synchronize(stream);
        }
        if let Some(stream) = routed_gpu.active_stream.as_ref() {
            routed_gpu.bind_thread_or_warn();
            let _ = routed_gpu.hip.stream_synchronize(stream);
        }
        free_opt(routed_gpu, &mut self.routed_x_rot);
        free_opt(routed_gpu, &mut self.routed_topk_indices);
        free_opt(routed_gpu, &mut self.routed_topk_weights);
        free_opt(routed_gpu, &mut self.routed_gate_batch);
        free_opt(routed_gpu, &mut self.routed_up_batch);
        free_opt(routed_gpu, &mut self.routed_rot_batch);
        free_opt(routed_gpu, &mut self.routed_down_expanded);
        free_opt(routed_gpu, &mut self.routed_partial);
        if let Some(signal) = self.signal_to_routed.take() {
            routed_gpu.bind_thread_or_warn();
            let _ = routed_gpu.hip.free(signal);
        }
        if let Some(signal) = self.signal_to_dense.take() {
            dense_gpu.bind_thread_or_warn();
            let _ = dense_gpu.hip.free(signal);
        }
        if let Some(event) = self.dense_attn_fork_event.take() {
            dense_gpu.bind_thread_or_warn();
            let _ = dense_gpu.hip.event_destroy(event);
        }
        if let Some(event) = self.dense_attn_join_event.take() {
            dense_gpu.bind_thread_or_warn();
            let _ = dense_gpu.hip.event_destroy(event);
        }
        if let Some(stream) = self.dense_attn_stream.take() {
            dense_gpu.bind_thread_or_warn();
            let _ = dense_gpu.hip.stream_destroy(stream);
        }
        if let Some(stream) = routed_gpu.active_stream.take() {
            routed_gpu.bind_thread_or_warn();
            let _ = routed_gpu.hip.stream_destroy(stream);
        }
        if let Some(stream) = dense_gpu.active_stream.take() {
            dense_gpu.bind_thread_or_warn();
            let _ = dense_gpu.hip.stream_destroy(stream);
        }
    }
}

impl DeepseekV4HeterogeneousModel {
    pub fn load(path: &Path, plan: DeepseekV4HeterogeneousLoadPlan) -> Result<Self, String> {
        Self::load_inner(path, plan, None, None)
    }

    pub fn load_verified(
        artifact: &DeepseekV4VerifiedArtifact,
        plan: DeepseekV4HeterogeneousLoadPlan,
    ) -> Result<Self, String> {
        Self::load_inner(artifact.path(), plan, None, Some(artifact))
    }

    /// Publish a replacement only after its entire two-device transaction has
    /// completed. If staging fails for any reason, `current` is never moved or
    /// mutated and remains available to its caller.
    pub fn replace_transactionally(
        current: &mut Option<Self>,
        path: &Path,
        plan: DeepseekV4HeterogeneousLoadPlan,
    ) -> Result<(), String> {
        let staged = Self::load(path, plan)?;
        let previous = current.replace(staged);
        drop(previous);
        Ok(())
    }

    pub fn replace_transactionally_verified(
        current: &mut Option<Self>,
        artifact: &DeepseekV4VerifiedArtifact,
        plan: DeepseekV4HeterogeneousLoadPlan,
    ) -> Result<(), String> {
        let staged = Self::load_verified(artifact, plan)?;
        let previous = current.replace(staged);
        drop(previous);
        Ok(())
    }

    #[doc(hidden)]
    pub fn load_with_fault(
        path: &Path,
        plan: DeepseekV4HeterogeneousLoadPlan,
        fault: DeepseekV4HeterogeneousFault,
    ) -> Result<Self, String> {
        Self::load_inner(path, plan, Some(fault), None)
    }

    #[doc(hidden)]
    pub fn load_verified_with_fault(
        artifact: &DeepseekV4VerifiedArtifact,
        plan: DeepseekV4HeterogeneousLoadPlan,
        fault: DeepseekV4HeterogeneousFault,
    ) -> Result<Self, String> {
        Self::load_inner(artifact.path(), plan, Some(fault), Some(artifact))
    }

    fn load_inner(
        path: &Path,
        plan: DeepseekV4HeterogeneousLoadPlan,
        fault: Option<DeepseekV4HeterogeneousFault>,
        verified: Option<&DeepseekV4VerifiedArtifact>,
    ) -> Result<Self, String> {
        ensure_harmonic_execution_admitted()?;
        if plan.prefill_max_batch == 0 {
            return Err("deepseek4 heterogeneous prefill_max_batch must be nonzero".into());
        }
        if plan.safety_margin_bytes < DEFAULT_SAFETY_MARGIN_BYTES {
            return Err(format!(
                "deepseek4 heterogeneous safety margin {} is below the 2 GiB contract",
                plan.safety_margin_bytes
            ));
        }

        let model_sha256 = if let Some(verified) = verified {
            verified.validate(path)?
        } else {
            let sha256 = crate::parent::manifest::sha256_file(path)?;
            if sha256 != MQ2R_0731_SHA256 {
                return Err(format!(
                    "deepseek4 heterogeneous artifact SHA mismatch: got {sha256}, expected {MQ2R_0731_SHA256}"
                ));
            }
            sha256
        };

        // Open the artifact exactly once. Both upload phases consume this one
        // index/file handle directly; no full-model owner or migration exists.
        let mut hfq = HfqFile::open(path)
            .map_err(|error| format!("deepseek4 heterogeneous open {}: {error}", path.display()))?;
        let mut config = <DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
        config.load_dspark = false;
        let projection = DeepseekV4::project_heterogeneous_gfx1100_gfx1151(&hfq, &config)?;
        // DeepseekV4State::new is allocation-free; all eager runtime storage is
        // the PBS inventory projected here.
        let dense_state_scratch_projected_bytes =
            PrefillBatchScratch::projected_allocation_bytes(&config, plan.prefill_max_batch)?;

        let (dense_device_id, routed_device_id) = plan.resolve_device_ids()?;
        let dense_gpu = Gpu::init_with_device(dense_device_id)
            .map_err(|error| format!("deepseek4 heterogeneous init dense device: {error}"))?;
        let routed_gpu = Gpu::init_with_device(routed_device_id)
            .map_err(|error| format!("deepseek4 heterogeneous init routed device: {error}"))?;
        if dense_gpu.arch != "gfx1100" || routed_gpu.arch != "gfx1151" {
            return Err(format!(
                "deepseek4 heterogeneous exact admission failed: dense={} dev {}, routed={} dev {}",
                dense_gpu.arch, dense_gpu.device_id, routed_gpu.arch, routed_gpu.device_id
            ));
        }
        let mut staging = DeepseekV4HeterogeneousStaging::new(dense_gpu, routed_gpu);
        let (dense_free_before, routed_free_before) = {
            let (dense_gpu, routed_gpu) = staging.devices_mut();
            dense_gpu
                .bind_thread()
                .map_err(|error| format!("deepseek4 heterogeneous bind dense: {error}"))?;
            let (dense_free, _) = dense_gpu
                .hip
                .get_vram_info()
                .map_err(|error| format!("deepseek4 heterogeneous dense meminfo: {error}"))?;
            routed_gpu
                .bind_thread()
                .map_err(|error| format!("deepseek4 heterogeneous bind routed: {error}"))?;
            let (routed_free, _) = routed_gpu
                .hip
                .get_vram_info()
                .map_err(|error| format!("deepseek4 heterogeneous routed meminfo: {error}"))?;
            (dense_free, routed_free)
        };

        let dense_minimum = projection
            .dense_bytes
            .checked_add(dense_state_scratch_projected_bytes)
            .and_then(|bytes| bytes.checked_add(plan.safety_margin_bytes))
            .ok_or_else(|| "deepseek4 heterogeneous dense preflight overflow".to_string())?;
        let routed_minimum = projection
            .routed_bytes
            .checked_add(plan.safety_margin_bytes)
            .ok_or_else(|| "deepseek4 heterogeneous routed preflight overflow".to_string())?;
        if dense_free_before < dense_minimum || routed_free_before < routed_minimum {
            return Err(format!(
                "deepseek4 heterogeneous preflight failed: dense free/weight+margin={dense_free_before}/{dense_minimum}, routed={routed_free_before}/{routed_minimum}"
            ));
        }

        let weights = {
            let (dense_gpu, routed_gpu) = staging.devices_mut();
            match fault {
                Some(fault) => DeepseekV4::load_weights_heterogeneous_gfx1100_gfx1151_with_fault(
                    &mut hfq, &config, dense_gpu, routed_gpu, fault,
                ),
                None => DeepseekV4::load_weights_heterogeneous_gfx1100_gfx1151(
                    &mut hfq, &config, dense_gpu, routed_gpu,
                ),
            }?
        };
        staging.weights = Some(weights);
        let ownership = staging.audit_weights()?;
        if fault == Some(DeepseekV4HeterogeneousFault::AfterOwnershipAudit) {
            return Err("deepseek4: injected heterogeneous failure after ownership audit".into());
        }

        staging.state = Some(DeepseekV4State::new(&config)?);
        if fault == Some(DeepseekV4HeterogeneousFault::AfterState) {
            return Err("deepseek4: injected heterogeneous failure after state".into());
        }

        staging.prefill = Some({
            let dense_gpu = staging
                .dense_gpu
                .as_mut()
                .ok_or_else(|| "deepseek4: dense staging device missing".to_string())?;
            PrefillBatchScratch::new(dense_gpu, &config, plan.prefill_max_batch)?
        });
        if fault == Some(DeepseekV4HeterogeneousFault::AfterScratch) {
            return Err("deepseek4: injected heterogeneous failure after scratch".into());
        }

        let (dense_free_after, routed_free_after) = {
            let (dense_gpu, routed_gpu) = staging.devices_mut();
            dense_gpu.bind_thread().map_err(|error| {
                format!("deepseek4 heterogeneous bind dense after load: {error}")
            })?;
            let (dense_free, _) = dense_gpu.hip.get_vram_info().map_err(|error| {
                format!("deepseek4 heterogeneous dense post-load meminfo: {error}")
            })?;
            routed_gpu.bind_thread().map_err(|error| {
                format!("deepseek4 heterogeneous bind routed after load: {error}")
            })?;
            let (routed_free, _) = routed_gpu.hip.get_vram_info().map_err(|error| {
                format!("deepseek4 heterogeneous routed post-load meminfo: {error}")
            })?;
            (dense_free, routed_free)
        };
        if dense_free_after < plan.safety_margin_bytes
            || routed_free_after < plan.safety_margin_bytes
        {
            return Err(format!(
                "deepseek4 heterogeneous post-load safety margin failed: dense={dense_free_after}, routed={routed_free_after}"
            ));
        }
        let dense_actual_bytes = dense_free_before.saturating_sub(dense_free_after);
        let routed_actual_bytes = routed_free_before.saturating_sub(routed_free_after);
        let dense_state_scratch_pool_bytes =
            dense_actual_bytes.saturating_sub(ownership.dense_bytes);
        let report = DeepseekV4HeterogeneousLoadReport {
            model_sha256,
            projection,
            dense_state_scratch_projected_bytes,
            ownership,
            dense_free_before,
            dense_free_after,
            routed_free_before,
            routed_free_after,
            dense_actual_bytes,
            routed_actual_bytes,
            dense_state_scratch_pool_bytes,
        };
        Ok(staging.publish(config, report))
    }

    pub fn audit_owners(&mut self) -> Result<DeepseekV4OwnershipAudit, String> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| "deepseek4: owner audit after weights were released".to_string())?;
        weights.audit_owners(&mut self.dense_gpu, &mut self.routed_gpu)
    }

    fn ensure_execution(&mut self) -> Result<(), String> {
        ensure_harmonic_execution_admitted()?;
        if self.execution.is_none() {
            self.execution = Some(DeepseekV4HeterogeneousExecution::new(
                &mut self.dense_gpu,
                &mut self.routed_gpu,
                &self.config,
            )?);
        }
        Ok(())
    }

    /// One exact direct-HIP heterogeneous token step. The caller owns greedy
    /// sampling and position advancement, matching `forward::decode_step`.
    pub fn decode_step(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, String> {
        self.decode_step_with_abort(token_id, position, &|| false)?
            .ok_or_else(|| "deepseek4 heterogeneous decode aborted without an abort source".into())
    }

    /// Cancellation-aware token step for the user-facing serve route.
    ///
    /// The callback is observed only before a layer starts or after the
    /// shared/routed fork has rejoined. An abort that arrives while either
    /// device is executing therefore drains to a safe cross-device boundary;
    /// it never abandons a peer signal or frees branch-owned storage in flight.
    pub fn decode_step_with_abort(
        &mut self,
        token_id: u32,
        position: u32,
        abort_requested: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, String> {
        self.ensure_execution()?;
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| "deepseek4 heterogeneous decode after weight release".to_string())?;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| "deepseek4 heterogeneous decode after state release".to_string())?;
        let execution = self
            .execution
            .as_mut()
            .ok_or_else(|| "deepseek4 heterogeneous execution missing".to_string())?;
        crate::forward::decode_step_heterogeneous(
            &self.config,
            weights,
            state,
            &mut self.dense_gpu,
            &mut self.routed_gpu,
            execution,
            token_id,
            position,
            abort_requested,
        )
    }

    /// Rewind the self-owned direct-HIP route for a fresh serving request.
    /// The split path does not yet expose prefix-cache reuse, so every request
    /// starts from the same zeroed recurrent/KV state as a fresh process.
    pub fn reset_for_request(&mut self) -> Result<(), String> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| "deepseek4 heterogeneous reset after state release".to_string())?;
        state.reset();
        state.zero_decode_caches(&mut self.dense_gpu);
        self.dense_gpu.invalidate_graph_state();
        self.routed_gpu.invalidate_graph_state();
        Ok(())
    }

    /// Fail-closed reset used after cancellation or an injected route error.
    /// Both owners are synchronized before the caller emits a terminal event,
    /// proving that no branch kernel, peer packet, or cache clear remains in
    /// flight when the request transaction is released.
    pub fn reset_for_request_attested(&mut self) -> Result<(), String> {
        self.reset_for_request()?;
        self.dense_gpu
            .bind_thread()
            .map_err(|error| format!("heterogeneous abort bind dense: {error}"))?;
        self.dense_gpu
            .hip
            .device_synchronize()
            .map_err(|error| format!("heterogeneous abort sync dense: {error:?}"))?;
        self.routed_gpu
            .bind_thread()
            .map_err(|error| format!("heterogeneous abort bind routed: {error}"))?;
        self.routed_gpu
            .hip
            .device_synchronize()
            .map_err(|error| format!("heterogeneous abort sync routed: {error:?}"))?;
        Ok(())
    }

    /// Explicit unload used by the G2 repeatability harness. `Drop` calls the
    /// same implementation, so early returns and normal scope exit converge.
    pub fn unload(mut self) {
        self.release();
    }

    fn release(&mut self) {
        if let Some(mut execution) = self.execution.take() {
            execution.release(&mut self.dense_gpu, &mut self.routed_gpu);
        }
        if let Some(prefill) = self.prefill.take() {
            prefill.free_gpu(&mut self.dense_gpu);
        }
        if let Some(state) = self.state.take() {
            state.free_gpu(&mut self.dense_gpu);
        }
        if let Some(weights) = self.weights.take() {
            weights.free_gpu(&mut self.dense_gpu, &mut self.routed_gpu);
        }
        self.dense_gpu.invalidate_weight_caches();
        self.dense_gpu.invalidate_graph_state();
        self.routed_gpu.invalidate_weight_caches();
        self.routed_gpu.invalidate_graph_state();
        self.dense_gpu.drain_pool();
        self.routed_gpu.drain_pool();
    }
}

impl Drop for DeepseekV4HeterogeneousModel {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_quarantine_is_stable_and_actionable() {
        let error = ensure_harmonic_execution_admitted().unwrap_err();
        assert_eq!(error, HARMONIC_EXECUTION_QUARANTINE);
        assert!(error.contains("reciprocal indefinite cross-device HIP waits"));
        assert!(error.contains("H2 fault-containment certification"));
    }

    #[test]
    fn harmonic_quarantine_precedes_path_and_gpu_access() {
        let missing = Path::new("/hipfire-h0-must-not-open-this-model");
        let error = match DeepseekV4HeterogeneousModel::load(
            missing,
            DeepseekV4HeterogeneousLoadPlan::default(),
        ) {
            Ok(_) => panic!("quarantined harmonic route unexpectedly loaded"),
            Err(error) => error,
        };
        assert_eq!(error, HARMONIC_EXECUTION_QUARANTINE);
    }
}
