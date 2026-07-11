//! Generic [`CompiledPlan`](crate::CompiledPlan) lowering onto the measured
//! two-queue public-AQL replay path.
//!
//! This layer is deliberately strict. Direct AQL is prepared only when every
//! kernel has an explicit kernarg ABI and artifact identity, every symbolic
//! resource/scalar is bound, and the compiled plan forms nonempty global
//! phases on exactly two lanes. Unsupported shapes fail closed so an embedding
//! runtime can retain HIP as its oracle/fallback.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::{
    AccessMode, ApiBoundary, ArtifactCatalog, CompiledPlan, DerivedVisibilityPlan, KernargAbi,
    KernelArg, KernelArtifactIdentity, NodeId, PreparedPlanInvalidation, PreparedPlanStamp,
    PreparedPlanState, ReplayBindingError, ReplayBindings, ReplayMode, VisibilityReason,
    derive_aql_visibility,
};
use redline_rocr::{
    FenceScope, GpuDevice, HeaderPolicy, KernargPool, Kernel, LaunchGeometry, RuntimeError,
};

use super::replay::TwoQueueDerivedPolicies;
use super::{
    GpuMultiQueueTiming, RecordedDispatch, ReplayError, TwoQueueBatchSubmission, TwoQueuePhase,
    TwoQueueSerializedBatchGraph,
};

/// Loader-resolved kernel plus the immutable identities required to prove that
/// a prepared AQL packet still targets the artifact recorded in the plan.
#[derive(Clone)]
struct AqlKernelBinding {
    kernel: Kernel,
    abi: KernargAbi,
    identity: KernelArtifactIdentity,
}

/// Explicit kernel catalog for generic AQL lowering.
///
/// The key is the HSA executable symbol consumed by `KernelLaunch` (normally a
/// `.kd` symbol). Registration checks loader metadata against the caller's
/// kernarg ABI before any queue is created.
#[derive(Clone, Default)]
pub struct AqlKernelCatalog {
    entries: BTreeMap<String, AqlKernelBinding>,
    identities: ArtifactCatalog,
}

impl AqlKernelCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        key: impl Into<String>,
        kernel: Kernel,
        abi: KernargAbi,
        identity: KernelArtifactIdentity,
    ) -> Result<(), GenericAqlError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(GenericAqlError::EmptyKernelKey);
        }
        let metadata = kernel.metadata();
        if metadata.kernarg_segment_size != abi.segment_size()
            || metadata.kernarg_segment_alignment != abi.segment_alignment()
        {
            return Err(GenericAqlError::LoaderAbiMismatch {
                kernel: key,
                loader_size: metadata.kernarg_segment_size,
                loader_alignment: metadata.kernarg_segment_alignment,
                recorded_size: abi.segment_size(),
                recorded_alignment: abi.segment_alignment(),
            });
        }
        if self.entries.contains_key(&key) {
            return Err(GenericAqlError::KernelAlreadyRegistered(key));
        }
        self.identities
            .insert(key.clone(), identity)
            .map_err(|_| GenericAqlError::EmptyKernelKey)?;
        self.entries.insert(
            key,
            AqlKernelBinding {
                kernel,
                abi,
                identity,
            },
        );
        Ok(())
    }

    pub fn artifact_catalog(&self) -> &ArtifactCatalog {
        &self.identities
    }
}

/// A generic plan lowered to the proven two-queue, one-doorbell-per-queue AQL
/// batch. The prepared graph owns queues, signals, packet images, kernels and
/// kernarg buffers; external resource pointees remain borrowed by contract.
pub struct PreparedAqlPlan {
    graph: TwoQueueSerializedBatchGraph,
    stamp: PreparedPlanStamp,
    phase_nodes: Vec<[Vec<NodeId>; 2]>,
    visibility: DerivedVisibilityPlan,
}

impl PreparedAqlPlan {
    /// Prepare `token_count` serialized copies of a two-lane compiled plan.
    ///
    /// Global dependency levels are lowered as phases. A phase must contain at
    /// least one dispatch on each lane; shapes that need a conservative generic
    /// graph are rejected so Auto can select HIP instead of silently changing
    /// the measured packet model.
    pub fn prepare(
        device: &GpuDevice,
        pool: &KernargPool,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        kernels: &AqlKernelCatalog,
        token_count: usize,
    ) -> Result<Self, GenericAqlError> {
        Self::prepare_with_boundary(
            device,
            pool,
            plan,
            bindings,
            kernels,
            token_count,
            ApiBoundary::default(),
        )
    }

    /// Prepare with an explicit host/HIP/AQL ownership boundary. This is the
    /// product cache/fence lever; the resulting policy remains inspectable via
    /// [`Self::visibility`].
    pub fn prepare_with_boundary(
        device: &GpuDevice,
        pool: &KernargPool,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        kernels: &AqlKernelCatalog,
        token_count: usize,
        boundary: ApiBoundary,
    ) -> Result<Self, GenericAqlError> {
        Self::prepare_internal(
            device,
            pool,
            plan,
            bindings,
            kernels,
            token_count,
            boundary,
            false,
        )
    }

    /// Prepare a diagnostic graph with a profiling signal on every dispatch.
    /// This provides an exact min-start/max-end GPU span for barrier-free tails
    /// but is intentionally separate from uninstrumented performance runs.
    pub fn prepare_profiled_all(
        device: &GpuDevice,
        pool: &KernargPool,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        kernels: &AqlKernelCatalog,
        token_count: usize,
        boundary: ApiBoundary,
    ) -> Result<Self, GenericAqlError> {
        Self::prepare_internal(
            device,
            pool,
            plan,
            bindings,
            kernels,
            token_count,
            boundary,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_internal(
        device: &GpuDevice,
        pool: &KernargPool,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        kernels: &AqlKernelCatalog,
        token_count: usize,
        boundary: ApiBoundary,
        profile_all_dispatches: bool,
    ) -> Result<Self, GenericAqlError> {
        if plan.replay_mode() != ReplayMode::TokenLatency {
            return Err(GenericAqlError::UnsupportedReplayMode(plan.replay_mode()));
        }
        if plan.lane_count().get() != 2 {
            return Err(GenericAqlError::UnsupportedLaneCount(
                plan.lane_count().get(),
            ));
        }
        if token_count == 0 {
            return Err(GenericAqlError::EmptyTokenBatch);
        }
        bindings.validate(plan)?;

        let (phases, phase_nodes) = lower_phases(device, pool, plan, bindings, kernels)?;
        let visibility = derive_aql_visibility(plan, boundary)?;
        let mut policies = lower_policies(plan, &phase_nodes, &visibility)?;
        policies.profile_all_dispatches = profile_all_dispatches;
        let base_packet_counts =
            TwoQueueSerializedBatchGraph::required_packet_counts(&phases, token_count)?;
        let consolidations = phases
            .len()
            .checked_mul(token_count)
            .ok_or(GenericAqlError::QueueSizeOverflow)?;
        let packet_counts = [
            base_packet_counts[0]
                .checked_add(consolidations)
                .ok_or(GenericAqlError::QueueSizeOverflow)?,
            base_packet_counts[1]
                .checked_add(consolidations)
                .ok_or(GenericAqlError::QueueSizeOverflow)?,
        ];
        let queue_size = queue_size_for(device, packet_counts)?;
        let graph = TwoQueueSerializedBatchGraph::create_derived(
            device,
            queue_size,
            token_count,
            phases,
            policies,
        )?;
        let stamp = PreparedPlanStamp::capture(plan, bindings)?;
        Ok(Self {
            graph,
            stamp,
            phase_nodes,
            visibility,
        })
    }

    pub fn queue_ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.graph.queue_ids()
    }

    pub fn batch_packet_counts(&self) -> [usize; 2] {
        self.graph.batch_packet_counts()
    }

    pub fn doorbell_writes_per_replay(&self) -> usize {
        self.graph.doorbell_writes_per_replay()
    }

    pub fn token_count(&self) -> usize {
        self.graph.token_count()
    }

    pub fn visibility(&self) -> &DerivedVisibilityPlan {
        &self.visibility
    }

    /// Read the completed first-to-last GPU timestamp span. This is available
    /// when the final dispatch on each lane is barriered and therefore covers
    /// every earlier packet in that queue (for example, the dependency DAG).
    pub fn gpu_batch_timing(&self) -> Result<GpuMultiQueueTiming, ReplayError> {
        self.graph.gpu_batch_timing()
    }

    pub fn validate_current(
        &self,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        kernels: &AqlKernelCatalog,
    ) -> Result<PreparedPlanState, PreparedPlanInvalidation> {
        self.stamp
            .validate(plan, bindings, kernels.artifact_catalog())
    }

    /// Patch dynamic scalar slots and rebound resource pointers into the
    /// graph's stable kernarg allocations, then advance its invalidation stamp.
    /// Artifact, ABI, layout, and plan changes remain hard invalidations.
    pub fn refresh_bindings(
        &mut self,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        kernels: &AqlKernelCatalog,
    ) -> Result<PreparedPlanState, GenericAqlError> {
        let state = self
            .stamp
            .validate(plan, bindings, kernels.artifact_catalog())
            .map_err(GenericAqlError::Invalidated)?;
        let phase_nodes = &self.phase_nodes;
        self.graph
            .patch_kernargs(|phase_index, lane, dispatch_index, kernarg| {
                let node = phase_nodes[phase_index][lane][dispatch_index];
                let dispatch = &plan.dispatches()[node.index() as usize];
                let abi = dispatch.launch().kernarg_abi().ok_or_else(|| {
                    GenericAqlError::MissingKernargAbi(dispatch.launch().kernel().to_owned())
                })?;
                pack_kernarg(dispatch.launch().arguments(), abi, bindings, kernarg)
            })?;
        self.stamp = PreparedPlanStamp::capture(plan, bindings)?;
        Ok(state)
    }

    /// Submit the prebuilt queue-local packet batches.
    ///
    /// # Safety
    ///
    /// Every external allocation represented by `ReplayBindings` at prepare
    /// time must remain live, GPU-accessible, and free of incompatible host or
    /// agent mutation until the ticket proves quiescence. On any error it must
    /// remain live through `PreparedAqlPlan` destruction.
    pub unsafe fn submit(&mut self) -> Result<TwoQueueBatchSubmission<'_>, ReplayError> {
        // SAFETY: the caller accepts the graph's external-pointee contract.
        unsafe { self.graph.submit() }
    }

    /// Submit and wait with the runtime's finite default timeout.
    ///
    /// # Safety
    ///
    /// The pointer lifetime contract is identical to [`Self::submit`].
    pub unsafe fn replay_and_wait(&mut self) -> Result<(), ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.graph.replay_and_wait() }
    }

    /// Submit and wait with an explicit finite timeout.
    ///
    /// # Safety
    ///
    /// The pointer lifetime contract is identical to [`Self::submit`].
    pub unsafe fn replay_and_wait_timeout(&mut self, timeout: Duration) -> Result<(), ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.graph.submit()? }.wait_timeout(timeout)
    }

    /// Refresh replay-time bindings, submit, and wait with a finite timeout.
    ///
    /// # Safety
    ///
    /// The pointer lifetime contract is identical to [`Self::submit`].
    pub unsafe fn replay_with_bindings_and_wait(
        &mut self,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        kernels: &AqlKernelCatalog,
        timeout: Duration,
    ) -> Result<PreparedPlanState, GenericAqlError> {
        let state = self.refresh_bindings(plan, bindings, kernels)?;
        // SAFETY: forwarded from this method's caller. The ticket is waited to
        // quiescence before returning.
        unsafe { self.graph.submit()? }.wait_timeout(timeout)?;
        Ok(state)
    }
}

fn lower_policies(
    plan: &CompiledPlan,
    phase_nodes: &[[Vec<NodeId>; 2]],
    visibility: &crate::DerivedVisibilityPlan,
) -> Result<TwoQueueDerivedPolicies, GenericAqlError> {
    let by_node = visibility
        .dispatches()
        .iter()
        .map(|policy| (policy.node(), policy))
        .collect::<BTreeMap<_, _>>();
    let mut first_dispatches = Vec::with_capacity(phase_nodes.len());
    let mut repeated_dispatches = Vec::with_capacity(phase_nodes.len());
    for phase in phase_nodes {
        let mut first_phase: [Vec<HeaderPolicy>; 2] = [Vec::new(), Vec::new()];
        let mut repeated_phase: [Vec<HeaderPolicy>; 2] = [Vec::new(), Vec::new()];
        for lane in 0..2 {
            for node in &phase[lane] {
                let policy = by_node
                    .get(node)
                    .ok_or(GenericAqlError::VisibilityNodeMissing(node.index()))?;
                let first = policy.header();
                let repeated_acquire = if policy
                    .acquire_reasons()
                    .iter()
                    .any(|reason| matches!(reason, VisibilityReason::RawConsumer { .. }))
                {
                    FenceScope::Agent
                } else {
                    FenceScope::None
                };
                first_phase[lane].push(first);
                repeated_phase[lane].push(HeaderPolicy {
                    barrier: first.barrier,
                    acquire: repeated_acquire,
                    release: first.release,
                });
            }
        }
        first_dispatches.push(first_phase);
        repeated_dispatches.push(repeated_phase);
    }
    let consolidation = visibility
        .lane_consolidations()
        .first()
        .ok_or(GenericAqlError::MissingLaneConsolidation)?
        .header();
    let cross_token_raw = has_cross_token_raw(plan);
    Ok(TwoQueueDerivedPolicies {
        first_dispatches,
        repeated_dispatches,
        consolidation: HeaderPolicy {
            release: if cross_token_raw {
                FenceScope::Agent
            } else {
                consolidation.release
            },
            ..consolidation
        },
        dependency: HeaderPolicy {
            acquire: if cross_token_raw {
                FenceScope::Agent
            } else {
                HeaderPolicy::TWO_QUEUE_DEPENDENCY.acquire
            },
            ..HeaderPolicy::TWO_QUEUE_DEPENDENCY
        },
        terminal: visibility.terminal().header(),
        profile_all_dispatches: false,
    })
}

/// A repeated token needs agent visibility only when a read can observe data
/// written by the preceding token before this token produces that region
/// itself. Ordinary output reuse (WAW) and same-token producer/consumer edges
/// need execution ordering but no cross-token cache fence.
fn has_cross_token_raw(plan: &CompiledPlan) -> bool {
    for (read_index, dispatch) in plan.dispatches().iter().enumerate() {
        for read in dispatch
            .accesses()
            .iter()
            .filter(|access| access.mode() == AccessMode::Read)
        {
            let written_earlier_this_token = plan.dispatches()[..read_index].iter().any(|prior| {
                prior.accesses().iter().any(|access| {
                    access.mode() == AccessMode::Write && access.region().overlaps(read.region())
                })
            });
            if written_earlier_this_token {
                continue;
            }
            if plan.dispatches().iter().any(|candidate| {
                candidate.accesses().iter().any(|access| {
                    access.mode() == AccessMode::Write && access.region().overlaps(read.region())
                })
            }) {
                return true;
            }
        }
    }
    false
}

fn lower_phases(
    device: &GpuDevice,
    pool: &KernargPool,
    plan: &CompiledPlan,
    bindings: &ReplayBindings,
    kernels: &AqlKernelCatalog,
) -> Result<(Vec<TwoQueuePhase>, Vec<[Vec<NodeId>; 2]>), GenericAqlError> {
    if plan.dispatches().is_empty() {
        return Err(GenericAqlError::EmptyPlan);
    }
    let mut levels = vec![0_usize; plan.dispatches().len()];
    let mut max_level = 0_usize;
    for dispatch in plan.dispatches() {
        let level = dispatch
            .dependencies()
            .iter()
            .map(|dependency| levels[dependency.index() as usize] + 1)
            .max()
            .unwrap_or(0);
        levels[dispatch.node().index() as usize] = level;
        max_level = max_level.max(level);
    }

    let mut phase_dispatches = (0..=max_level)
        .map(|_| [Vec::new(), Vec::new()])
        .collect::<Vec<[Vec<RecordedDispatch>; 2]>>();
    let mut phase_nodes = (0..=max_level)
        .map(|_| [Vec::new(), Vec::new()])
        .collect::<Vec<[Vec<NodeId>; 2]>>();
    for dispatch in plan.dispatches() {
        let lane = dispatch.lane().0;
        let level = levels[dispatch.node().index() as usize];
        let launch = dispatch.launch();
        let registered = kernels
            .entries
            .get(launch.kernel())
            .ok_or_else(|| GenericAqlError::KernelNotRegistered(launch.kernel().to_owned()))?;
        let recorded_abi = launch
            .kernarg_abi()
            .ok_or_else(|| GenericAqlError::MissingKernargAbi(launch.kernel().to_owned()))?;
        if recorded_abi != &registered.abi {
            return Err(GenericAqlError::RegisteredAbiChanged(
                launch.kernel().to_owned(),
            ));
        }
        let recorded_identity = launch
            .artifact_identity()
            .ok_or_else(|| GenericAqlError::MissingArtifactIdentity(launch.kernel().to_owned()))?;
        if recorded_identity != registered.identity {
            return Err(GenericAqlError::ArtifactIdentityChanged(
                launch.kernel().to_owned(),
            ));
        }
        let mut kernarg = pool.allocate_for(registered.kernel.metadata())?;
        pack_kernarg(launch.arguments(), recorded_abi, bindings, &mut kernarg)?;
        let grid = launch.grid();
        let block = launch.block();
        let workgroup = [
            u16::try_from(block.x).map_err(|_| GenericAqlError::BlockDimensionTooLarge(0))?,
            u16::try_from(block.y).map_err(|_| GenericAqlError::BlockDimensionTooLarge(1))?,
            u16::try_from(block.z).map_err(|_| GenericAqlError::BlockDimensionTooLarge(2))?,
        ];
        let geometry = LaunchGeometry::from_hip_workgroups([grid.x, grid.y, grid.z], workgroup)
            .map_err(ReplayError::from)?;
        device.validate_geometry(geometry)?;
        let recorded = RecordedDispatch::new(lane, registered.kernel.clone(), geometry, kernarg)?
            .with_dynamic_group_bytes(launch.dynamic_shared_bytes())?;
        phase_dispatches[level][lane].push(recorded);
        phase_nodes[level][lane].push(dispatch.node());
    }

    let phases = phase_dispatches
        .into_iter()
        .enumerate()
        .map(|(phase, lanes)| {
            if lanes[0].is_empty() || lanes[1].is_empty() {
                return Err(GenericAqlError::EmptyPhaseLane { phase });
            }
            let [lane_zero, lane_one] = lanes;
            TwoQueuePhase::new(lane_zero, lane_one).map_err(GenericAqlError::Replay)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((phases, phase_nodes))
}

fn pack_kernarg(
    arguments: &[KernelArg],
    abi: &KernargAbi,
    bindings: &ReplayBindings,
    output: &mut redline_rocr::KernargBuffer,
) -> Result<(), GenericAqlError> {
    abi.validate_arguments(arguments)
        .map_err(GenericAqlError::KernargAbi)?;
    if output.len() != abi.segment_size() as usize {
        return Err(GenericAqlError::KernargBufferSize {
            expected: abi.segment_size() as usize,
            actual: output.len(),
        });
    }
    let bytes = output.as_mut_bytes();
    bytes.fill(0);
    for (argument, field) in arguments.iter().zip(abi.fields()) {
        let value: Vec<u8> = match argument {
            KernelArg::Scalar(value) => value.to_vec(),
            KernelArg::ScalarSlot { slot, size } => {
                let value = bindings
                    .scalar(*slot)
                    .ok_or(ReplayBindingError::ScalarNotBound { slot: *slot })?;
                if value.len() != *size as usize {
                    return Err(ReplayBindingError::ScalarSize {
                        slot: *slot,
                        expected: *size,
                        actual: value.len(),
                    }
                    .into());
                }
                value.to_vec()
            }
            KernelArg::Resource {
                resource,
                byte_offset,
            } => {
                let binding =
                    bindings
                        .resource(*resource)
                        .ok_or(ReplayBindingError::ResourceNotBound {
                            resource: *resource,
                        })?;
                if *byte_offset >= binding.size() {
                    return Err(GenericAqlError::ResourceArgumentOutOfBounds {
                        resource: resource.index(),
                        offset: *byte_offset,
                        bound: binding.size(),
                    });
                }
                let offset = usize::try_from(*byte_offset).map_err(|_| {
                    GenericAqlError::ResourcePointerOverflow {
                        resource: resource.index(),
                    }
                })?;
                let address = (binding.base().as_ptr() as usize)
                    .checked_add(offset)
                    .ok_or(GenericAqlError::ResourcePointerOverflow {
                        resource: resource.index(),
                    })?;
                (address as u64).to_le_bytes().to_vec()
            }
        };
        let start = field.offset() as usize;
        let end = start + field.size() as usize;
        bytes[start..end].copy_from_slice(&value);
    }
    Ok(())
}

fn queue_size_for(device: &GpuDevice, packet_counts: [usize; 2]) -> Result<u32, GenericAqlError> {
    let required = u32::try_from(packet_counts[0].max(packet_counts[1]))
        .map_err(|_| GenericAqlError::QueueSizeOverflow)?;
    let minimum = *device.queue_size_range().start();
    let maximum = *device.queue_size_range().end();
    let requested = required
        .max(minimum)
        .checked_next_power_of_two()
        .ok_or(GenericAqlError::QueueSizeOverflow)?;
    if requested > maximum {
        return Err(GenericAqlError::QueueCapacity { required, maximum });
    }
    Ok(requested)
}

#[derive(Debug, thiserror::Error)]
pub enum GenericAqlError {
    #[error("kernel key is empty")]
    EmptyKernelKey,
    #[error("kernel {0:?} is already registered")]
    KernelAlreadyRegistered(String),
    #[error("kernel {0:?} is not registered")]
    KernelNotRegistered(String),
    #[error("kernel {0:?} has no explicit kernarg ABI")]
    MissingKernargAbi(String),
    #[error("kernel {0:?} has no artifact identity")]
    MissingArtifactIdentity(String),
    #[error("kernel {0:?} registered kernarg ABI differs from the compiled plan")]
    RegisteredAbiChanged(String),
    #[error("kernel {0:?} registered artifact identity differs from the compiled plan")]
    ArtifactIdentityChanged(String),
    #[error(
        "kernel {kernel:?} loader ABI size/alignment {loader_size}/{loader_alignment} differs from recorded {recorded_size}/{recorded_alignment}"
    )]
    LoaderAbiMismatch {
        kernel: String,
        loader_size: u32,
        loader_alignment: u32,
        recorded_size: u32,
        recorded_alignment: u32,
    },
    #[error("generic direct AQL currently requires exactly two lanes, got {0}")]
    UnsupportedLaneCount(usize),
    #[error("generic direct AQL currently supports token-latency plans, got {0:?}")]
    UnsupportedReplayMode(ReplayMode),
    #[error("compiled plan contains no dispatches")]
    EmptyPlan,
    #[error("token batch is empty")]
    EmptyTokenBatch,
    #[error("dependency phase {phase} leaves one AQL lane empty")]
    EmptyPhaseLane { phase: usize },
    #[error("work-group dimension {0} exceeds the AQL u16 field")]
    BlockDimensionTooLarge(usize),
    #[error("kernarg ABI is invalid: {0}")]
    KernargAbi(#[source] crate::KernargAbiError),
    #[error("kernarg allocation has {actual} bytes, expected {expected}")]
    KernargBufferSize { expected: usize, actual: usize },
    #[error("resource {resource} argument offset {offset} exceeds binding size {bound}")]
    ResourceArgumentOutOfBounds {
        resource: u32,
        offset: u64,
        bound: u64,
    },
    #[error("resource {resource} pointer arithmetic overflowed")]
    ResourcePointerOverflow { resource: u32 },
    #[error("queue packet count does not fit u32")]
    QueueSizeOverflow,
    #[error("queue needs {required} packets but device maximum is {maximum}")]
    QueueCapacity { required: u32, maximum: u32 },
    #[error("derived visibility policy omitted node index {0}")]
    VisibilityNodeMissing(u32),
    #[error("derived visibility policy omitted queue consolidation")]
    MissingLaneConsolidation,
    #[error("replay bindings are invalid: {0}")]
    Bindings(#[from] ReplayBindingError),
    #[error("ROCr operation failed: {0}")]
    Runtime(#[from] RuntimeError),
    #[error("AQL replay construction failed: {0}")]
    Replay(#[from] ReplayError),
    #[error("AQL visibility derivation failed: {0}")]
    Visibility(#[from] crate::VisibilityError),
    #[error("prepared AQL plan was invalidated: {0}")]
    Invalidated(#[from] PreparedPlanInvalidation),
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{Access, CompileOptions, Dim3, KernelLaunch, Recorder};

    fn header(policy: HeaderPolicy, packet_type: u16) -> u16 {
        packet_type
            | (u16::from(policy.barrier) << 8)
            | ((policy.acquire as u16) << 9)
            | ((policy.release as u16) << 11)
    }

    #[test]
    fn generic_policy_keeps_system_acquire_only_at_batch_entry() {
        let mut recorder = Recorder::new();
        let weights_id = recorder.resource("weights", 64).unwrap();
        let outputs_id = recorder.resource("outputs", 256).unwrap();
        let weights = recorder.region(weights_id, 0, 64).unwrap();
        for index in 0..4 {
            let output = recorder.region(outputs_id, index * 64, 64).unwrap();
            recorder
                .dispatch(
                    KernelLaunch::new(
                        format!("kernel-{index}"),
                        Dim3::x(1).unwrap(),
                        Dim3::x(32).unwrap(),
                    )
                    .unwrap(),
                    [Access::read(weights), Access::write(output)],
                )
                .unwrap();
        }
        let plan = recorder
            .compile(CompileOptions::new(
                NonZeroUsize::new(2).unwrap(),
                ReplayMode::TokenLatency,
            ))
            .unwrap();
        let mut nodes = [[Vec::new(), Vec::new()]];
        for dispatch in plan.dispatches() {
            nodes[0][dispatch.lane().0].push(dispatch.node());
        }
        assert!(nodes[0].iter().all(|lane| lane.len() == 2));
        let visibility = derive_aql_visibility(&plan, ApiBoundary::default()).unwrap();
        let policies = lower_policies(&plan, &nodes, &visibility).unwrap();

        for lane in 0..2 {
            assert_eq!(header(policies.first_dispatches[0][lane][0], 2), 0x0402);
            assert_eq!(header(policies.first_dispatches[0][lane][1], 2), 0x0002);
            assert_eq!(header(policies.repeated_dispatches[0][lane][0], 2), 0x0002);
            assert_eq!(header(policies.repeated_dispatches[0][lane][1], 2), 0x0002);
        }
        assert_eq!(header(policies.consolidation, 3), 0x0103);
        assert_eq!(header(policies.dependency, 3), 0x0103);
        assert_eq!(header(policies.terminal, 3), 0x1103);
    }

    #[test]
    fn repeated_read_before_write_adds_agent_visibility_between_tokens() {
        let mut recorder = Recorder::new();
        let state_id = recorder.resource("state", 64).unwrap();
        let state = recorder.region(state_id, 0, 64).unwrap();
        let read = recorder
            .dispatch(
                KernelLaunch::new("read", Dim3::x(1).unwrap(), Dim3::x(32).unwrap()).unwrap(),
                [Access::read(state)],
            )
            .unwrap();
        let write = recorder
            .dispatch(
                KernelLaunch::new("write", Dim3::x(1).unwrap(), Dim3::x(32).unwrap()).unwrap(),
                [Access::write(state)],
            )
            .unwrap();
        recorder.depends_on(write, read).unwrap();
        let plan = recorder
            .compile(CompileOptions::new(
                NonZeroUsize::new(2).unwrap(),
                ReplayMode::TokenLatency,
            ))
            .unwrap();
        assert!(has_cross_token_raw(&plan));
    }
}
