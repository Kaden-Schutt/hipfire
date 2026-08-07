// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Exact-owner parent half of the DeepSeek V4 harmonic AR route.
//!
//! The current process owns only gfx1100 dense weights/state and a mapped
//! source ring. The long-lived gfx1151 child owns every routed expert. No HIP
//! object, device pointer, queue signal, or ordinal crosses the process seam.

use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::time::Duration;

use hip_bridge::{Event, Stream};
use hipfire_config::{Deepseek4ComputePlacement, DeviceSelector};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::arch::{DeepseekV4HarmonicResidencyProjection, DeepseekV4HeterogeneousProjection};
use crate::deepseek4::{
    DeepseekV4Config, DeepseekV4DenseWeights, DeepseekV4HarmonicReplicaWeights, DeepseekV4State,
};
use crate::forward::PrefillBatchScratch;
use crate::harmonic::{
    HarmonicContract, HarmonicExpertResidencyPlan, HarmonicOwner, HarmonicPackedExpertRoute,
    HARMONIC_EXPERT_COUNT, HARMONIC_HIDDEN_SIZE, HARMONIC_LAYER_COUNT,
    HARMONIC_MOE_INTERMEDIATE_SIZE, HARMONIC_TOP_K,
};
use crate::harmonic_ipc::{harmonic_monotonic_tick, HarmonicGpuMapping, HarmonicSharedRing};
use crate::harmonic_process::{
    HarmonicExpertWorkerProcess, HarmonicExpertWorkerReady, HarmonicExpertWorkerSpec,
};
use crate::heterogeneous::{
    DeepseekV4HeterogeneousLoadPlan, DeepseekV4VerifiedArtifact, DEFAULT_SAFETY_MARGIN_BYTES,
};
use crate::DeepseekV4;

const DEFAULT_LAYER_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Clone, Debug)]
pub struct DeepseekV4HarmonicLoadPlan {
    pub placement: Deepseek4ComputePlacement,
    pub worker_executable: PathBuf,
    pub runtime_dir: PathBuf,
    pub prefill_max_batch: usize,
    pub safety_margin_bytes: usize,
    pub startup_timeout: Duration,
    pub control_timeout: Duration,
    pub exit_timeout: Duration,
    pub layer_timeout: Duration,
    pub residency_plan: Option<HarmonicExpertResidencyPlan>,
}

impl DeepseekV4HarmonicLoadPlan {
    pub fn new(worker_executable: PathBuf, runtime_dir: PathBuf) -> Self {
        Self {
            placement: Deepseek4ComputePlacement::DenseExpertSplit {
                dense: DeviceSelector::ExactArch("gfx1100".into()),
                experts: DeviceSelector::ExactArch("gfx1151".into()),
            },
            worker_executable,
            runtime_dir,
            prefill_max_batch: 1024,
            safety_margin_bytes: DEFAULT_SAFETY_MARGIN_BYTES,
            startup_timeout: Duration::from_secs(600),
            control_timeout: Duration::from_secs(2),
            exit_timeout: Duration::from_secs(5),
            layer_timeout: DEFAULT_LAYER_TIMEOUT,
            residency_plan: None,
        }
    }

    pub fn with_residency_plan(mut self, residency_plan: HarmonicExpertResidencyPlan) -> Self {
        self.residency_plan = Some(residency_plan);
        self
    }

    fn validate(&self) -> Result<(), String> {
        if !self.worker_executable.is_file() {
            return Err(format!(
                "deepseek4 harmonic expert worker is not a file: {}",
                self.worker_executable.display()
            ));
        }
        if self.prefill_max_batch == 0 {
            return Err("deepseek4 harmonic prefill_max_batch must be nonzero".to_owned());
        }
        if self.safety_margin_bytes < DEFAULT_SAFETY_MARGIN_BYTES {
            return Err(format!(
                "deepseek4 harmonic safety margin {} is below the 2 GiB contract",
                self.safety_margin_bytes
            ));
        }
        if self.startup_timeout.is_zero()
            || self.control_timeout.is_zero()
            || self.exit_timeout.is_zero()
            || self.layer_timeout.is_zero()
        {
            return Err("deepseek4 harmonic timeouts must be nonzero".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV4HarmonicLoadReport {
    pub model_sha256: String,
    pub projection: DeepseekV4HeterogeneousProjection,
    pub dense_pci_bus_id: String,
    pub expert: HarmonicExpertWorkerReady,
    pub dense_free_before: usize,
    pub dense_free_after: usize,
    pub hotset_projection: Option<DeepseekV4HarmonicResidencyProjection>,
}

/// Always-on aggregate timing for the harmonic diagnostic route. These are
/// host-observed synchronization buckets, not kernel-profile claims: route
/// sync absorbs the gfx1100 prefix through route publication, while expert
/// wait spans release-publication through gfx1151 completion visibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeepseekV4HarmonicTiming {
    pub tokens: u64,
    pub layers: u64,
    pub layer_wall_ns: u64,
    pub route_sync_ns: u64,
    pub route_sync_max_ns: u64,
    pub expert_wait_ns: u64,
    pub expert_wait_max_ns: u64,
    pub publish_cpu_ns: u64,
    pub join_enqueue_cpu_ns: u64,
}

impl DeepseekV4HarmonicTiming {
    fn add_ns(total: &mut u64, elapsed: Duration) -> u64 {
        let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        *total = total.saturating_add(nanos);
        nanos
    }
}

pub(crate) struct DeepseekV4HarmonicExecution {
    pub(crate) dense_attn_stream: Option<Stream>,
    pub(crate) dense_attn_fork_event: Option<Event>,
    pub(crate) dense_attn_join_event: Option<Event>,
    pub(crate) transfer_stream: Option<Stream>,
    pub(crate) route_ready_event: Option<Event>,
    pub(crate) ring: HarmonicSharedRing,
    pub(crate) mapping: Option<HarmonicGpuMapping>,
    pub(crate) worker: Option<HarmonicExpertWorkerProcess>,
    pub(crate) contract: HarmonicContract,
    pub(crate) source_generation: u64,
    pub(crate) layer_timeout: Duration,
    pub(crate) route_ids: [u32; HARMONIC_TOP_K],
    pub(crate) route_weight_bits: [u32; HARMONIC_TOP_K],
    pub(crate) epoch: u64,
    pub(crate) timing: DeepseekV4HarmonicTiming,
    pub(crate) residency_plan: Option<HarmonicExpertResidencyPlan>,
    pub(crate) compact_index_map: Option<GpuTensor>,
    pub(crate) local_topk_indices: Option<GpuTensor>,
    pub(crate) local_topk_count: Option<GpuTensor>,
    pub(crate) local_gate_batch: Option<GpuTensor>,
    pub(crate) local_up_batch: Option<GpuTensor>,
    pub(crate) local_rot_batch: Option<GpuTensor>,
    pub(crate) local_down_expanded: Option<GpuTensor>,
    pub(crate) slot_sources: Option<GpuTensor>,
    pub(crate) packed_route: Option<HarmonicPackedExpertRoute>,
    ring_path: PathBuf,
    control_socket: PathBuf,
}

impl DeepseekV4HarmonicExecution {
    fn new(
        gpu: &mut Gpu,
        ring: HarmonicSharedRing,
        mapping: HarmonicGpuMapping,
        worker: HarmonicExpertWorkerProcess,
        contract: HarmonicContract,
        residency_plan: Option<HarmonicExpertResidencyPlan>,
        layer_timeout: Duration,
        ring_path: PathBuf,
        control_socket: PathBuf,
    ) -> Result<Self, String> {
        if gpu.active_stream.is_some() {
            return Err("deepseek4 harmonic gfx1100 primary stream already claimed".to_owned());
        }
        gpu.bind_thread()
            .map_err(|error| format!("deepseek4 harmonic bind gfx1100 execution: {error}"))?;
        let mut execution = Self {
            dense_attn_stream: None,
            dense_attn_fork_event: None,
            dense_attn_join_event: None,
            transfer_stream: None,
            route_ready_event: None,
            ring,
            mapping: Some(mapping),
            worker: Some(worker),
            contract,
            source_generation: contract.source_allocation_generation,
            layer_timeout,
            route_ids: [0; HARMONIC_TOP_K],
            route_weight_bits: [0; HARMONIC_TOP_K],
            epoch: 0,
            timing: DeepseekV4HarmonicTiming::default(),
            residency_plan,
            compact_index_map: None,
            local_topk_indices: None,
            local_topk_count: None,
            local_gate_batch: None,
            local_up_batch: None,
            local_rot_batch: None,
            local_down_expanded: None,
            slot_sources: None,
            packed_route: None,
            ring_path,
            control_socket,
        };
        let result = (|| {
            gpu.active_stream = Some(
                gpu.hip
                    .stream_create()
                    .map_err(|error| format!("deepseek4 harmonic primary stream: {error}"))?,
            );
            execution.dense_attn_stream = Some(
                gpu.hip
                    .stream_create()
                    .map_err(|error| format!("deepseek4 harmonic attention stream: {error}"))?,
            );
            execution.transfer_stream = Some(
                gpu.hip
                    .stream_create()
                    .map_err(|error| format!("deepseek4 harmonic transfer stream: {error}"))?,
            );
            execution.dense_attn_fork_event = Some(
                gpu.hip
                    .event_create()
                    .map_err(|error| format!("deepseek4 harmonic attention fork: {error}"))?,
            );
            execution.dense_attn_join_event = Some(
                gpu.hip
                    .event_create()
                    .map_err(|error| format!("deepseek4 harmonic attention join: {error}"))?,
            );
            execution.route_ready_event = Some(
                gpu.hip
                    .event_create()
                    .map_err(|error| format!("deepseek4 harmonic route-ready event: {error}"))?,
            );
            if execution.residency_plan.is_some() {
                let compact_index_table = execution
                    .residency_plan
                    .as_ref()
                    .unwrap()
                    .compact_index_table();
                let compact_index_bytes = unsafe {
                    std::slice::from_raw_parts(
                        compact_index_table.as_ptr().cast::<u8>(),
                        compact_index_table.len() * std::mem::size_of::<u32>(),
                    )
                };
                let compact_index_map = gpu
                    .alloc_tensor(
                        &[
                            HARMONIC_LAYER_COUNT as usize,
                            HARMONIC_EXPERT_COUNT as usize,
                        ],
                        DType::F32,
                    )
                    .map_err(|error| format!("deepseek4 harmonic compact map: {error}"))?;
                gpu.hip
                    .memcpy_htod(&compact_index_map.buf, compact_index_bytes)
                    .map_err(|error| format!("deepseek4 harmonic compact map upload: {error}"))?;
                execution.compact_index_map = Some(compact_index_map);
                execution.local_topk_indices = Some(
                    gpu.alloc_tensor(&[HARMONIC_TOP_K], DType::F32)
                        .map_err(|error| format!("deepseek4 harmonic local indices: {error}"))?,
                );
                execution.local_topk_count = Some(
                    gpu.alloc_tensor(&[1], DType::F32)
                        .map_err(|error| format!("deepseek4 harmonic local count: {error}"))?,
                );
                execution.local_gate_batch = Some(
                    gpu.alloc_tensor(
                        &[HARMONIC_TOP_K, HARMONIC_MOE_INTERMEDIATE_SIZE],
                        DType::F32,
                    )
                    .map_err(|error| format!("deepseek4 harmonic local gate: {error}"))?,
                );
                execution.local_up_batch = Some(
                    gpu.alloc_tensor(
                        &[HARMONIC_TOP_K, HARMONIC_MOE_INTERMEDIATE_SIZE],
                        DType::F32,
                    )
                    .map_err(|error| format!("deepseek4 harmonic local up: {error}"))?,
                );
                execution.local_rot_batch = Some(
                    gpu.alloc_tensor(
                        &[HARMONIC_TOP_K, HARMONIC_MOE_INTERMEDIATE_SIZE],
                        DType::F32,
                    )
                    .map_err(|error| format!("deepseek4 harmonic local rotation: {error}"))?,
                );
                execution.local_down_expanded = Some(
                    gpu.alloc_tensor(&[HARMONIC_TOP_K, HARMONIC_HIDDEN_SIZE], DType::F32)
                        .map_err(|error| format!("deepseek4 harmonic local down: {error}"))?,
                );
                execution.slot_sources = Some(
                    gpu.alloc_tensor(&[HARMONIC_TOP_K], DType::F32)
                        .map_err(|error| format!("deepseek4 harmonic slot sources: {error}"))?,
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            let _ = execution.release(gpu);
            return Err(error);
        }
        Ok(execution)
    }

    pub(crate) fn record_route_sync(&mut self, elapsed: Duration) {
        let nanos = DeepseekV4HarmonicTiming::add_ns(&mut self.timing.route_sync_ns, elapsed);
        self.timing.route_sync_max_ns = self.timing.route_sync_max_ns.max(nanos);
    }

    pub(crate) fn record_expert_wait(&mut self, elapsed: Duration) {
        let nanos = DeepseekV4HarmonicTiming::add_ns(&mut self.timing.expert_wait_ns, elapsed);
        self.timing.expert_wait_max_ns = self.timing.expert_wait_max_ns.max(nanos);
    }

    pub(crate) fn record_publish_cpu(&mut self, elapsed: Duration) {
        DeepseekV4HarmonicTiming::add_ns(&mut self.timing.publish_cpu_ns, elapsed);
    }

    pub(crate) fn record_join_enqueue_cpu(&mut self, elapsed: Duration) {
        DeepseekV4HarmonicTiming::add_ns(&mut self.timing.join_enqueue_cpu_ns, elapsed);
    }

    pub(crate) fn record_layer(&mut self, elapsed: Duration) {
        self.timing.layers = self.timing.layers.saturating_add(1);
        DeepseekV4HarmonicTiming::add_ns(&mut self.timing.layer_wall_ns, elapsed);
    }

    pub(crate) fn record_token(&mut self) {
        self.timing.tokens = self.timing.tokens.saturating_add(1);
    }

    pub(crate) fn next_epoch(&mut self) -> Result<u64, String> {
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| "deepseek4 harmonic epoch exhausted".to_owned())?;
        Ok(self.epoch)
    }

    pub(crate) fn isolate_worker(&mut self, cause: String) -> String {
        let Some(mut worker) = self.worker.take() else {
            return format!("{cause}; harmonic expert worker already isolated");
        };
        match worker.terminate_and_isolate(cause.clone()) {
            Ok(receipt) => format!(
                "{cause}; expert pid={} exit={} isolated_slots={}",
                receipt.pid, receipt.exit_status, receipt.isolated_slots
            ),
            Err(error) => format!("{cause}; expert isolation failed: {error}"),
        }
    }

    fn release(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let mut errors = Vec::new();
        gpu.bind_thread_or_warn();
        for stream in [
            gpu.active_stream.as_ref(),
            self.dense_attn_stream.as_ref(),
            self.transfer_stream.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if let Err(error) = gpu.hip.stream_synchronize(stream) {
                errors.push(format!("synchronize gfx1100 stream: {error}"));
            }
        }
        if let Some(worker) = self.worker.take() {
            if let Err(error) = worker.shutdown_and_isolate() {
                errors.push(error);
            }
        }
        if let Some(mapping) = self.mapping.as_mut() {
            if let Err(error) = mapping.unregister(&gpu.hip) {
                errors.push(format!("unregister gfx1100 harmonic mapping: {error}"));
            }
        }
        self.mapping.take();
        for tensor in [
            self.compact_index_map.take(),
            self.local_topk_indices.take(),
            self.local_topk_count.take(),
            self.local_gate_batch.take(),
            self.local_up_batch.take(),
            self.local_rot_batch.take(),
            self.local_down_expanded.take(),
            self.slot_sources.take(),
        ]
        .into_iter()
        .flatten()
        {
            if let Err(error) = gpu.free_tensor(tensor) {
                errors.push(format!("free gfx1100 harmonic split scratch: {error}"));
            }
        }
        for event in [
            self.route_ready_event.take(),
            self.dense_attn_fork_event.take(),
            self.dense_attn_join_event.take(),
        ]
        .into_iter()
        .flatten()
        {
            if let Err(error) = gpu.hip.event_destroy(event) {
                errors.push(format!("destroy gfx1100 harmonic event: {error}"));
            }
        }
        for stream in [
            self.transfer_stream.take(),
            self.dense_attn_stream.take(),
            gpu.active_stream.take(),
        ]
        .into_iter()
        .flatten()
        {
            if let Err(error) = gpu.hip.stream_destroy(stream) {
                errors.push(format!("destroy gfx1100 harmonic stream: {error}"));
            }
        }
        let _ = fs::remove_file(&self.control_socket);
        let _ = fs::remove_file(&self.ring_path);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

pub struct DeepseekV4HarmonicModel {
    pub dense_gpu: Gpu,
    pub config: DeepseekV4Config,
    pub weights: Option<DeepseekV4DenseWeights>,
    pub state: Option<DeepseekV4State>,
    pub prefill: Option<PrefillBatchScratch>,
    pub replicas: Option<DeepseekV4HarmonicReplicaWeights>,
    execution: Option<DeepseekV4HarmonicExecution>,
    pub report: DeepseekV4HarmonicLoadReport,
}

impl DeepseekV4HarmonicModel {
    pub fn load_verified(
        artifact: &DeepseekV4VerifiedArtifact,
        plan: DeepseekV4HarmonicLoadPlan,
    ) -> Result<Self, String> {
        plan.validate()?;
        fs::create_dir_all(&plan.runtime_dir).map_err(|error| {
            format!(
                "create harmonic runtime directory {}: {error}",
                plan.runtime_dir.display()
            )
        })?;
        let model_sha256 = artifact.validate(artifact.path())?;
        let artifact_receipt = artifact.receipt()?;
        let mut hfq = HfqFile::open(artifact.path()).map_err(|error| {
            format!(
                "deepseek4 harmonic open {}: {error}",
                artifact.path().display()
            )
        })?;
        let mut config = <DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
        config.load_dspark = false;
        let projection = DeepseekV4::project_heterogeneous_gfx1100_gfx1151(&hfq, &config)?;
        let placement = DeepseekV4HeterogeneousLoadPlan {
            placement: plan.placement.clone(),
            prefill_max_batch: plan.prefill_max_batch,
            safety_margin_bytes: plan.safety_margin_bytes,
        };
        let (dense_device_id, expert_device_id) = placement.resolve_device_ids()?;
        let discovery = hip_bridge::HipRuntime::load()
            .map_err(|error| format!("deepseek4 harmonic HIP discovery: {error}"))?;
        let dense_pci_bus_id = discovery
            .device_pci_bus_id(dense_device_id)
            .map_err(|error| format!("deepseek4 harmonic dense PCI identity: {error}"))?;
        let expert_pci_bus_id = discovery
            .device_pci_bus_id(expert_device_id)
            .map_err(|error| format!("deepseek4 harmonic expert PCI identity: {error}"))?;

        let mut dense_gpu = Gpu::init_with_device(dense_device_id)
            .map_err(|error| format!("deepseek4 harmonic init dense device: {error}"))?;
        HarmonicOwner::DenseGfx1100
            .validate_arch(&dense_gpu.arch)
            .map_err(|error| error.to_string())?;
        dense_gpu
            .bind_thread()
            .map_err(|error| format!("deepseek4 harmonic bind dense load: {error}"))?;
        let (dense_free_before, _) = dense_gpu
            .hip
            .get_vram_info()
            .map_err(|error| format!("deepseek4 harmonic dense preflight: {error}"))?;
        let hotset_projection = plan
            .residency_plan
            .as_ref()
            .map(|residency| {
                DeepseekV4::project_harmonic_hot_experts_gfx1100(
                    &hfq, &config, &dense_gpu, residency,
                )
            })
            .transpose()?;
        let replica_bytes = hotset_projection
            .as_ref()
            .map(|projection| projection.exact_bytes)
            .unwrap_or(0);
        let split_scratch_bytes = if hotset_projection.is_some() {
            (HARMONIC_TOP_K
                * (3 * HARMONIC_MOE_INTERMEDIATE_SIZE + HARMONIC_HIDDEN_SIZE)
                * std::mem::size_of::<f32>()
                + 2 * HARMONIC_TOP_K * std::mem::size_of::<u32>()) as u64
        } else {
            0
        };
        let replica_bytes = usize::try_from(replica_bytes).map_err(|_| {
            "deepseek4 harmonic replica projection exceeds address space".to_owned()
        })?;
        let split_scratch_bytes = usize::try_from(split_scratch_bytes)
            .map_err(|_| "deepseek4 harmonic split scratch exceeds address space".to_owned())?;
        let dense_projected = projection
            .dense_bytes
            .checked_add(PrefillBatchScratch::projected_allocation_bytes(
                &config,
                plan.prefill_max_batch,
            )?)
            .and_then(|bytes| bytes.checked_add(replica_bytes))
            .and_then(|bytes| bytes.checked_add(split_scratch_bytes))
            .and_then(|bytes| bytes.checked_add(plan.safety_margin_bytes))
            .ok_or_else(|| "deepseek4 harmonic dense preflight overflow".to_owned())?;
        if dense_free_before < dense_projected {
            return Err(format!(
                "deepseek4 harmonic dense preflight failed: free={dense_free_before}, required={dense_projected}"
            ));
        }

        let mut weights = Some(DeepseekV4::load_weights_harmonic_dense_gfx1100(
            &mut hfq,
            &config,
            &mut dense_gpu,
        )?);
        let mut replicas = match plan.residency_plan.as_ref() {
            Some(residency) => match DeepseekV4::load_weights_harmonic_hot_experts_gfx1100(
                &mut hfq,
                &config,
                &mut dense_gpu,
                residency,
            ) {
                Ok(replicas) => Some(replicas),
                Err(error) => {
                    weights.take().unwrap().free_gpu(&mut dense_gpu);
                    return Err(error);
                }
            },
            None => None,
        };
        let mut state = match DeepseekV4State::new(&config) {
            Ok(state) => Some(state),
            Err(error) => {
                if let Some(replicas) = replicas.take() {
                    replicas.free_gpu(&mut dense_gpu);
                }
                weights.take().unwrap().free_gpu(&mut dense_gpu);
                return Err(error);
            }
        };
        let mut prefill =
            match PrefillBatchScratch::new(&mut dense_gpu, &config, plan.prefill_max_batch) {
                Ok(prefill) => Some(prefill),
                Err(error) => {
                    state.take().unwrap().free_gpu(&mut dense_gpu);
                    if let Some(replicas) = replicas.take() {
                        replicas.free_gpu(&mut dense_gpu);
                    }
                    weights.take().unwrap().free_gpu(&mut dense_gpu);
                    return Err(error);
                }
            };
        let dense_free_after = match dense_gpu
            .bind_thread()
            .map_err(|error| format!("deepseek4 harmonic bind dense post-load: {error}"))
            .and_then(|_| {
                dense_gpu
                    .hip
                    .get_vram_info()
                    .map(|(free, _)| free)
                    .map_err(|error| format!("deepseek4 harmonic dense post-load: {error}"))
            }) {
            Ok(free) => free,
            Err(error) => {
                prefill.take().unwrap().free_gpu(&mut dense_gpu);
                state.take().unwrap().free_gpu(&mut dense_gpu);
                if let Some(replicas) = replicas.take() {
                    replicas.free_gpu(&mut dense_gpu);
                }
                weights.take().unwrap().free_gpu(&mut dense_gpu);
                return Err(error);
            }
        };

        let nonce = harmonic_monotonic_tick()
            .map_err(|error| format!("deepseek4 harmonic runtime nonce: {error}"))?;
        let source_generation = nonce.max(1);
        let destination_generation = nonce.rotate_left(17).max(1);
        let contract = plan.residency_plan.as_ref().map_or_else(
            || HarmonicContract::frozen(source_generation, destination_generation),
            |residency| {
                HarmonicContract::hotset(
                    source_generation,
                    destination_generation,
                    residency.identity(),
                )
            },
        );
        let prefix = format!("ds4-harmonic-{}-{nonce}", std::process::id());
        let ring_path = plan.runtime_dir.join(format!("{prefix}.ring"));
        let control_socket = plan.runtime_dir.join(format!("{prefix}.sock"));
        let ring_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&ring_path)
            .map_err(|error| format!("create harmonic ring {}: {error}", ring_path.display()))?;
        let mut ring = HarmonicSharedRing::create_data_plane(&ring_file, contract)
            .map_err(|error| format!("initialize harmonic ring: {error}"))?;
        let mut mapping = Some(
            HarmonicGpuMapping::register(&mut ring, &dense_gpu.hip)
                .map_err(|error| format!("register gfx1100 harmonic mapping: {error}"))?,
        );
        let worker = HarmonicExpertWorkerProcess::spawn(HarmonicExpertWorkerSpec {
            executable: plan.worker_executable.clone(),
            model: artifact.path().to_path_buf(),
            artifact_receipt,
            pci_bus_id: expert_pci_bus_id,
            ring: ring_path.clone(),
            control_socket: control_socket.clone(),
            allocation_generation: destination_generation,
            first_epoch: 1,
            startup_timeout: plan.startup_timeout,
            control_timeout: plan.control_timeout,
            exit_timeout: plan.exit_timeout,
        });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                if let Some(mapping) = mapping.as_mut() {
                    let _ = mapping.unregister(&dense_gpu.hip);
                }
                let _ = fs::remove_file(&control_socket);
                let _ = fs::remove_file(&ring_path);
                prefill.take().unwrap().free_gpu(&mut dense_gpu);
                state.take().unwrap().free_gpu(&mut dense_gpu);
                if let Some(replicas) = replicas.take() {
                    replicas.free_gpu(&mut dense_gpu);
                }
                weights.take().unwrap().free_gpu(&mut dense_gpu);
                return Err(error);
            }
        };
        let expert = worker.ready().clone();
        let execution = DeepseekV4HarmonicExecution::new(
            &mut dense_gpu,
            ring,
            mapping.take().unwrap(),
            worker,
            contract,
            plan.residency_plan.clone(),
            plan.layer_timeout,
            ring_path,
            control_socket,
        );
        let execution = match execution {
            Ok(execution) => execution,
            Err(error) => {
                prefill.take().unwrap().free_gpu(&mut dense_gpu);
                state.take().unwrap().free_gpu(&mut dense_gpu);
                if let Some(replicas) = replicas.take() {
                    replicas.free_gpu(&mut dense_gpu);
                }
                weights.take().unwrap().free_gpu(&mut dense_gpu);
                return Err(error);
            }
        };
        Ok(Self {
            dense_gpu,
            config,
            weights,
            state,
            prefill,
            replicas,
            execution: Some(execution),
            report: DeepseekV4HarmonicLoadReport {
                model_sha256,
                projection,
                dense_pci_bus_id,
                expert,
                dense_free_before,
                dense_free_after,
                hotset_projection,
            },
        })
    }

    pub fn decode_step(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, String> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| "deepseek4 harmonic decode after weight release".to_owned())?;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| "deepseek4 harmonic decode after state release".to_owned())?;
        let execution = self
            .execution
            .as_mut()
            .ok_or_else(|| "deepseek4 harmonic execution unavailable".to_owned())?;
        crate::forward::decode_step_harmonic(
            &self.config,
            &weights.inner,
            state,
            &mut self.dense_gpu,
            execution,
            self.replicas.as_ref(),
            token_id,
            position,
        )
    }

    pub fn reset_timing(&mut self) -> Result<(), String> {
        let execution = self
            .execution
            .as_mut()
            .ok_or_else(|| "deepseek4 harmonic execution unavailable".to_owned())?;
        execution.timing = DeepseekV4HarmonicTiming::default();
        Ok(())
    }

    pub fn timing(&self) -> Result<DeepseekV4HarmonicTiming, String> {
        self.execution
            .as_ref()
            .map(|execution| execution.timing)
            .ok_or_else(|| "deepseek4 harmonic execution unavailable".to_owned())
    }

    pub fn shutdown(mut self) -> Result<(), String> {
        self.release()
    }

    fn release(&mut self) -> Result<(), String> {
        let mut errors = Vec::new();
        if let Some(mut execution) = self.execution.take() {
            if let Err(error) = execution.release(&mut self.dense_gpu) {
                errors.push(error);
            }
        }
        if let Some(prefill) = self.prefill.take() {
            prefill.free_gpu(&mut self.dense_gpu);
        }
        if let Some(state) = self.state.take() {
            state.free_gpu(&mut self.dense_gpu);
        }
        if let Some(replicas) = self.replicas.take() {
            replicas.free_gpu(&mut self.dense_gpu);
        }
        if let Some(weights) = self.weights.take() {
            weights.free_gpu(&mut self.dense_gpu);
        }
        self.dense_gpu.invalidate_weight_caches();
        self.dense_gpu.invalidate_graph_state();
        self.dense_gpu.drain_pool();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

impl Drop for DeepseekV4HarmonicModel {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            eprintln!("deepseek4 harmonic release: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roles_are_portable_exact_arch_selectors() {
        let plan =
            DeepseekV4HarmonicLoadPlan::new(PathBuf::from("worker"), PathBuf::from("runtime"));
        assert_eq!(
            plan.placement,
            Deepseek4ComputePlacement::DenseExpertSplit {
                dense: DeviceSelector::ExactArch("gfx1100".into()),
                experts: DeviceSelector::ExactArch("gfx1151".into()),
            }
        );
    }
}
