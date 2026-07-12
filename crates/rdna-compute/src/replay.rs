// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Default-off integration gate for Redline record/replay.
//!
//! This module records the central HIP launch surface during warmup and owns
//! the fail-closed selection state. It deliberately does not reinterpret
//! `void**` arguments: a model adapter must supply explicit resource accesses
//! and a kernarg ABI to `redline-dispatch` before installing a prepared plan.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use redline_dispatch::aql::{
    load_symbols, BatchFencePolicy, Executable, Gfx12Pm4CommandBuffer, GpuBatchTiming, GpuSelector,
    HeaderPolicy, KernargBuffer, KernargPool, Kernel, LaunchGeometry, RecordedDispatch, Runtime,
    SingleQueueBatchGraph, SingleQueuePm4Ib,
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

impl ReplayTransport {
    fn from_env() -> Self {
        match std::env::var("HIPFIRE_REPLAY_TRANSPORT")
            .unwrap_or_else(|_| "aql".to_owned())
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

    fn from_env() -> Self {
        let value = std::env::var("HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY")
            .unwrap_or_else(|_| "required-only".to_owned());
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
    matches!(
        previous,
        "repeat_interleave_qk_f32" | "rope_partial_halfsplit_f32"
    ) || matches!(
        current,
        "repeat_interleave_qk_f32" | "rope_partial_halfsplit_f32"
    )
}

fn conservative_mid_acquire_except(previous: &str, current: &str, excluded: Option<&str>) -> bool {
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
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )
    )
}

impl ReplayBackendRequest {
    fn from_env() -> Self {
        match std::env::var("HIPFIRE_REPLAY_BACKEND")
            .unwrap_or_else(|_| "hip".to_owned())
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "hip" | "off" => Self::Hip,
            "shadow" => Self::Shadow,
            "auto" => Self::Auto,
            value => {
                eprintln!("WARNING: unknown HIPFIRE_REPLAY_BACKEND={value:?}; falling back to hip");
                Self::Hip
            }
        }
    }
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
    /// Exact naturally-aligned, tail-padded bytes passed through HIP's
    /// contiguous `extra` launch ABI. Pointer values intentionally remain
    /// opaque here; the model adapter owns their lifetime contract.
    pub kernarg: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCaptureSummary {
    pub launch_count: usize,
    pub unique_kernel_count: usize,
    pub sequence_hash: u64,
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

pub struct PreparedPm4Replay {
    graph: SingleQueuePm4Ib,
    // Kernels retain their HSA executables and kernargs retain every pointer
    // programmed into the immutable indirect buffer.
    _kernels: Vec<Kernel>,
    kernargs: Vec<KernargBuffer>,
    dynamic_gdn_frames: Vec<usize>,
    dispatch_count: usize,
    command_dwords: u32,
}

impl PreparedPm4Replay {
    /// # Safety
    ///
    /// Every pointer captured in the immutable explicit kernarg prefixes must
    /// still refer to the same live Hipfire allocation and model instance.
    pub unsafe fn replay_and_wait(&mut self) -> Result<(), String> {
        for dispatch in &self.dynamic_gdn_frames {
            let frame = crate::norm::reserve_gdn_requant_frames(1);
            let bytes = self.kernargs[*dispatch].as_mut_bytes();
            bytes
                .get_mut(76..80)
                .ok_or_else(|| "PM4 GDN kernarg is too short for frame patch".to_owned())?
                .copy_from_slice(&frame.to_ne_bytes());
        }
        // SAFETY: forwarded from the caller that owns the model allocations.
        unsafe { self.graph.replay_and_wait() }.map_err(|error| error.to_string())
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    pub fn command_dwords(&self) -> u32 {
        self.command_dwords
    }

    pub fn queue_id(&self) -> u64 {
        self.graph.queue_id()
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
}

impl ReplayController {
    pub fn from_env() -> Self {
        let request = ReplayBackendRequest::from_env();
        let manual = std::env::var("HIPFIRE_REPLAY_MANUAL_CAPTURE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "on"))
            .unwrap_or(false);
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
            transport: ReplayTransport::from_env(),
            pm4_mid_acquire_policy: Pm4MidAcquirePolicy::from_env(),
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
        }
    }

    pub fn new_armed(request: ReplayBackendRequest) -> Self {
        let mut controller = Self::new(request);
        if request != ReplayBackendRequest::Hip {
            controller.state = ReplayState::Armed;
        }
        controller
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
            if launch.kernel == "gated_delta_net_q8_fast" {
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

    /// Lower a captured prefix to one retained GFX12 PM4 indirect buffer.
    /// The initial diagnostic form serializes every dispatch boundary. The
    /// coherence gate can then remove only boundaries proven independent.
    pub fn prepare_pm4_prefix(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
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
        let pool = KernargPool::discover(&device).map_err(|error| error.to_string())?;
        let mut executables = BTreeMap::<PathBuf, Executable>::new();
        let mut resolved = BTreeMap::<(PathBuf, String), Kernel>::new();
        let mut kernels = Vec::with_capacity(prefix);
        let mut kernargs = Vec::with_capacity(prefix);
        let mut geometries = Vec::with_capacity(prefix);
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
            if launch.kernel == "gated_delta_net_q8_fast" {
                if metadata.kernarg_segment_size < 80 {
                    return Err(format!(
                        "{symbol}: loader kernarg is too short for dynamic frame binding"
                    ));
                }
                dynamic_gdn_frames.push(kernargs.len());
            }
            kernels.push(kernel);
            kernargs.push(kernarg);
            geometries.push(geometry);
        }

        let mut commands = Gfx12Pm4CommandBuffer::new();
        commands.acquire_system();
        for index in 0..prefix {
            if index != 0 {
                let previous = self.recorded[index - 1].kernel.as_str();
                let current = self.recorded[index].kernel.as_str();
                if !independent_sibling(previous, current) {
                    commands.wait_compute_idle();
                }
                if self
                    .pm4_mid_acquire_policy
                    .acquire_between(previous, current)
                {
                    commands.acquire_system();
                }
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
        let command_dwords = commands.len_dwords();
        let graph = SingleQueuePm4Ib::create(&device, &pool, &commands)
            .map_err(|error| error.to_string())?;
        let queue_id = graph.queue_id();
        self.prepared_pm4 = Some(PreparedPm4Replay {
            graph,
            _kernels: kernels,
            kernargs,
            dynamic_gdn_frames,
            dispatch_count: prefix,
            command_dwords,
        });
        self.state = ReplayState::Ready;
        Ok((prefix, command_dwords, queue_id))
    }

    /// # Safety
    ///
    /// The captured model allocations and all pointed-to buffers must still be
    /// live and in the same binding layout.
    pub unsafe fn replay_linear_aql(&mut self) -> Result<GpuBatchTiming, String> {
        let prepared = self
            .prepared
            .as_mut()
            .ok_or_else(|| "no prepared AQL replay".to_owned())?;
        // SAFETY: forwarded from the model owner.
        unsafe { prepared.replay_and_wait() }
    }

    /// # Safety
    ///
    /// The captured model allocations and all pointed-to buffers must still be
    /// live and in the same binding layout.
    pub unsafe fn replay_pm4(&mut self) -> Result<(), String> {
        let prepared = self
            .prepared_pm4
            .as_mut()
            .ok_or_else(|| "no prepared PM4 replay".to_owned())?;
        // SAFETY: forwarded from the model owner.
        unsafe { prepared.replay_and_wait() }
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
        }
        ReplayCaptureSummary {
            launch_count: self.recorded.len(),
            unique_kernel_count,
            sequence_hash: hash,
        }
    }

    pub(crate) fn record_hip_launch(
        &mut self,
        kernel: &str,
        artifact: Option<PathBuf>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
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
            kernarg: kernarg.to_vec(),
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
    fn moe_shared_down_and_routed_gate_up_are_independent_siblings() {
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
        ));
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
        assert!(
            Pm4MidAcquirePolicy::Conservative
                .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32")
        );
        assert!(
            !Pm4MidAcquirePolicy::EntryOnly
                .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32")
        );
        assert!(!Pm4MidAcquirePolicy::Conservative.acquire_between("rmsnorm_f32", "gemv_hfq4g256"));
        assert!(
            !Pm4MidAcquirePolicy::WithoutRope
                .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32")
        );
        assert!(
            Pm4MidAcquirePolicy::WithoutRope
                .acquire_between("repeat_interleave_qk_f32", "rope_partial_halfsplit_f32")
        );
        assert!(
            !Pm4MidAcquirePolicy::WithoutMqRotate.acquire_between("mq_rotate_x", "gemv_hfq4g256")
        );
        assert!(
            Pm4MidAcquirePolicy::RequiredOnly
                .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32")
        );
        assert!(
            !Pm4MidAcquirePolicy::RequiredOnly
                .acquire_between("fused_silu_mul_mq_rotate", "gemv_hfq4g256")
        );
        assert_eq!(Pm4MidAcquirePolicy::from_value("invalid"), None);
    }

    #[test]
    fn default_hip_never_records_or_routes() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Hip);
        controller.record_hip_launch("k", None, [1; 3], [32, 1, 1], 0, &[]);
        assert!(controller.recorded_launches().is_empty());
        assert!(!controller.should_route_aql());
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
