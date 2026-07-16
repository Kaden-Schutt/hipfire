use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::num::NonZeroUsize;

use sha2::{Digest, Sha256};

use crate::backend::{BeginReplay, DispatchBackend, DispatchRequest, EndReplay};
use crate::recorder::Recorder;
use crate::{
    Access, AccessMode, DeviceRegion, KernargAbiError, KernelArg, KernelArtifactIdentity,
    KernelLaunch, NodeId, ReplayMode, ReplayToken, ScalarSlotId,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LaneId(pub usize);

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PlanFingerprint([u8; 32]);

impl PlanFingerprint {
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn from_bytes_for_test(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl fmt::Debug for PlanFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0[..8] {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str("…")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompileOptions {
    lane_count: NonZeroUsize,
    replay_mode: ReplayMode,
}

impl CompileOptions {
    pub fn new(lane_count: NonZeroUsize, replay_mode: ReplayMode) -> Self {
        Self {
            lane_count,
            replay_mode,
        }
    }

    pub fn lanes(lane_count: usize, replay_mode: ReplayMode) -> Option<Self> {
        NonZeroUsize::new(lane_count).map(|lane_count| Self::new(lane_count, replay_mode))
    }

    pub fn lane_count(self) -> NonZeroUsize {
        self.lane_count
    }

    pub fn replay_mode(self) -> ReplayMode {
        self.replay_mode
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HazardKind {
    /// An earlier-in-record-order write conflicts with a later read.
    ReadAfterWrite,
    /// An earlier-in-record-order read conflicts with a later write.
    WriteAfterRead,
    /// Two writes conflict.
    WriteAfterWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Hazard {
    pub first: NodeId,
    pub second: NodeId,
    pub kind: HazardKind,
    pub overlap: DeviceRegion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedResource {
    id: crate::ResourceId,
    label: String,
    size: u64,
}

impl PlannedResource {
    pub fn id(&self) -> crate::ResourceId {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedDispatch {
    node: NodeId,
    lane: LaneId,
    launch: KernelLaunch,
    accesses: Vec<Access>,
    dependencies: Vec<NodeId>,
    estimated_start: u64,
    estimated_end: u64,
}

impl PlannedDispatch {
    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn lane(&self) -> LaneId {
        self.lane
    }

    pub fn launch(&self) -> &KernelLaunch {
        &self.launch
    }

    pub fn accesses(&self) -> &[Access] {
        &self.accesses
    }

    pub fn dependencies(&self) -> &[NodeId] {
        &self.dependencies
    }

    pub fn estimated_start(&self) -> u64 {
        self.estimated_start
    }

    pub fn estimated_end(&self) -> u64 {
        self.estimated_end
    }
}

/// A validated DAG assigned to deterministic logical lanes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledPlan {
    resources: Vec<PlannedResource>,
    dispatches: Vec<PlannedDispatch>,
    terminals: Vec<NodeId>,
    lane_count: NonZeroUsize,
    replay_mode: ReplayMode,
    fingerprint: PlanFingerprint,
}

impl CompiledPlan {
    pub(crate) fn compile(
        recorder: &Recorder,
        options: CompileOptions,
    ) -> Result<Self, CompileError> {
        validate_kernel_contracts(recorder)?;
        let order = topological_order(recorder)?;
        let reachable = reachability(recorder, &order);
        let hazards = unordered_hazards(recorder, &reachable);
        if !hazards.is_empty() {
            return Err(CompileError::UnorderedHazards(hazards));
        }

        let lane_count = options.lane_count.get();
        let mut lane_available = vec![0_u64; lane_count];
        let mut node_finish = vec![0_u64; recorder.nodes.len()];
        let mut dispatches = Vec::with_capacity(order.len());

        for node in order {
            let entry = &recorder.nodes[node.index as usize];
            let dependencies = recorder
                .edges
                .iter()
                .filter_map(|(source, target)| (*target == node).then_some(*source))
                .collect::<Vec<_>>();
            let dependency_finish = dependencies
                .iter()
                .map(|dependency| node_finish[dependency.index as usize])
                .max()
                .unwrap_or(0);
            let work = u64::from(entry.launch.estimated_work().get());
            let (lane, start, end) = lane_available
                .iter()
                .enumerate()
                .map(|(lane, available)| {
                    let start = (*available).max(dependency_finish);
                    (lane, start, start.saturating_add(work))
                })
                .min_by_key(|(lane, start, end)| (*end, *start, *lane))
                .expect("lane count is nonzero");
            lane_available[lane] = end;
            node_finish[node.index as usize] = end;
            dispatches.push(PlannedDispatch {
                node,
                lane: LaneId(lane),
                launch: entry.launch.clone(),
                accesses: entry.accesses.clone(),
                dependencies,
                estimated_start: start,
                estimated_end: end,
            });
        }

        let terminals = recorder
            .nodes
            .iter()
            .filter(|node| !recorder.edges.iter().any(|(source, _)| *source == node.id))
            .map(|node| node.id)
            .collect();

        let resources = recorder
            .resources
            .iter()
            .enumerate()
            .map(|(index, resource)| PlannedResource {
                id: crate::ResourceId {
                    owner: recorder.owner,
                    index: index as u32,
                },
                label: resource.label.clone(),
                size: resource.size,
            })
            .collect();

        let mut plan = Self {
            resources,
            dispatches,
            terminals,
            lane_count: options.lane_count,
            replay_mode: options.replay_mode,
            fingerprint: PlanFingerprint([0; 32]),
        };
        plan.fingerprint = fingerprint(&plan);
        Ok(plan)
    }

    pub fn dispatches(&self) -> &[PlannedDispatch] {
        &self.dispatches
    }

    pub fn resources(&self) -> &[PlannedResource] {
        &self.resources
    }

    pub fn terminal_nodes(&self) -> &[NodeId] {
        &self.terminals
    }

    pub fn lane_count(&self) -> NonZeroUsize {
        self.lane_count
    }

    pub fn replay_mode(&self) -> ReplayMode {
        self.replay_mode
    }

    pub const fn fingerprint(&self) -> PlanFingerprint {
        self.fingerprint
    }

    pub fn scalar_slots(&self) -> BTreeMap<ScalarSlotId, u32> {
        let mut slots = BTreeMap::new();
        for dispatch in &self.dispatches {
            for argument in dispatch.launch.arguments() {
                if let KernelArg::ScalarSlot { slot, size } = argument {
                    slots.insert(*slot, *size);
                }
            }
        }
        slots
    }

    pub fn artifact_identities(&self) -> BTreeMap<&str, KernelArtifactIdentity> {
        let mut artifacts = BTreeMap::new();
        for dispatch in &self.dispatches {
            if let Some(identity) = dispatch.launch.artifact_identity() {
                artifacts.insert(dispatch.launch.kernel(), identity);
            }
        }
        artifacts
    }

    pub fn replay<B: DispatchBackend>(
        &self,
        backend: &mut B,
        token: ReplayToken,
    ) -> Result<B::Completion, B::Error> {
        backend.begin_replay(BeginReplay {
            token,
            mode: self.replay_mode,
            lane_count: self.lane_count.get(),
        })?;

        let mut signals: Vec<Option<B::Signal>> = vec![None; self.dispatches.len()];
        for dispatch in &self.dispatches {
            let dependency_signals = dispatch
                .dependencies
                .iter()
                .map(|dependency| {
                    signals[dependency.index as usize]
                        .as_ref()
                        .expect("topological order guarantees dependency submission")
                        .clone()
                })
                .collect::<Vec<_>>();
            let signal = backend.dispatch(DispatchRequest {
                token,
                node: dispatch.node,
                lane: dispatch.lane,
                launch: &dispatch.launch,
                accesses: &dispatch.accesses,
                dependency_signals: &dependency_signals,
            })?;
            signals[dispatch.node.index as usize] = Some(signal);
        }

        let terminal_signals = self
            .terminals
            .iter()
            .map(|node| {
                signals[node.index as usize]
                    .as_ref()
                    .expect("terminal node was submitted")
                    .clone()
            })
            .collect::<Vec<_>>();
        backend.end_replay(EndReplay {
            token,
            mode: self.replay_mode,
            terminal_signals: &terminal_signals,
        })
    }
}

fn validate_kernel_contracts(recorder: &Recorder) -> Result<(), CompileError> {
    let mut slots = BTreeMap::<ScalarSlotId, u32>::new();
    let mut kernels = BTreeMap::new();
    for node in &recorder.nodes {
        let launch = &node.launch;
        if let Some(abi) = launch.kernarg_abi() {
            abi.validate_arguments(launch.arguments())
                .map_err(|cause| CompileError::InvalidKernargAbi {
                    kernel: launch.kernel().to_owned(),
                    cause,
                })?;
        }
        for argument in launch.arguments() {
            if let KernelArg::ScalarSlot { slot, size } = argument {
                if let Some(previous) = slots.insert(*slot, *size) {
                    if previous != *size {
                        return Err(CompileError::ConflictingScalarSlotSize {
                            slot: *slot,
                            first: previous,
                            second: *size,
                        });
                    }
                }
            }
        }
        let identity = (
            launch.kernarg_abi().map(|abi| abi.hash()),
            launch.artifact_identity(),
        );
        if let Some(previous) = kernels.insert(launch.kernel().to_owned(), identity) {
            if previous != identity {
                return Err(CompileError::ConflictingKernelIdentity {
                    kernel: launch.kernel().to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn fingerprint(plan: &CompiledPlan) -> PlanFingerprint {
    let mut hash = Sha256::new();
    hash.update(b"redline-compiled-plan-v1\0");
    put_u64(&mut hash, plan.lane_count.get() as u64);
    match plan.replay_mode {
        ReplayMode::TokenLatency => hash.update([0]),
        ReplayMode::Throughput {
            max_tokens_in_flight,
        } => {
            hash.update([1]);
            put_u64(&mut hash, max_tokens_in_flight.get() as u64);
        }
    }
    put_u64(&mut hash, plan.resources.len() as u64);
    for resource in &plan.resources {
        put_u32(&mut hash, resource.id.index());
        put_bytes(&mut hash, resource.label.as_bytes());
        put_u64(&mut hash, resource.size);
    }
    put_u64(&mut hash, plan.dispatches.len() as u64);
    for dispatch in &plan.dispatches {
        put_u32(&mut hash, dispatch.node.index());
        put_u64(&mut hash, dispatch.lane.0 as u64);
        put_bytes(&mut hash, dispatch.launch.kernel().as_bytes());
        put_dim3(&mut hash, dispatch.launch.grid());
        put_dim3(&mut hash, dispatch.launch.block());
        put_u32(&mut hash, dispatch.launch.dynamic_shared_bytes());
        put_u32(&mut hash, dispatch.launch.estimated_work().get());
        match dispatch.launch.kernarg_abi() {
            Some(abi) => {
                hash.update([1]);
                hash.update(abi.hash().digest().as_bytes());
            }
            None => hash.update([0]),
        }
        match dispatch.launch.artifact_identity() {
            Some(identity) => {
                hash.update([1]);
                hash.update(identity.code_object().as_bytes());
                hash.update(identity.symbol_text().as_bytes());
                put_u64(&mut hash, identity.generation());
            }
            None => hash.update([0]),
        }
        put_u64(&mut hash, dispatch.launch.arguments().len() as u64);
        for argument in dispatch.launch.arguments() {
            match argument {
                KernelArg::Scalar(bytes) => {
                    hash.update([0]);
                    put_bytes(&mut hash, bytes);
                }
                KernelArg::ScalarSlot { slot, size } => {
                    hash.update([1]);
                    put_u32(&mut hash, slot.index());
                    put_u32(&mut hash, *size);
                }
                KernelArg::Resource {
                    resource,
                    byte_offset,
                } => {
                    hash.update([2]);
                    put_u32(&mut hash, resource.index());
                    put_u64(&mut hash, *byte_offset);
                }
            }
        }
        put_u64(&mut hash, dispatch.accesses.len() as u64);
        for access in &dispatch.accesses {
            hash.update([match access.mode() {
                AccessMode::Read => 0,
                AccessMode::Write => 1,
            }]);
            put_u32(&mut hash, access.region().resource().index());
            put_u64(&mut hash, access.region().offset());
            put_u64(&mut hash, access.region().len());
        }
        put_u64(&mut hash, dispatch.dependencies.len() as u64);
        for dependency in &dispatch.dependencies {
            put_u32(&mut hash, dependency.index());
        }
    }
    put_u64(&mut hash, plan.terminals.len() as u64);
    for terminal in &plan.terminals {
        put_u32(&mut hash, terminal.index());
    }
    PlanFingerprint(hash.finalize().into())
}

fn put_dim3(hash: &mut Sha256, dimensions: crate::Dim3) {
    put_u32(hash, dimensions.x);
    put_u32(hash, dimensions.y);
    put_u32(hash, dimensions.z);
}

fn put_u32(hash: &mut Sha256, value: u32) {
    hash.update(value.to_le_bytes());
}

fn put_u64(hash: &mut Sha256, value: u64) {
    hash.update(value.to_le_bytes());
}

fn put_bytes(hash: &mut Sha256, bytes: &[u8]) {
    put_u64(hash, bytes.len() as u64);
    hash.update(bytes);
}

fn topological_order(recorder: &Recorder) -> Result<Vec<NodeId>, CompileError> {
    let mut indegree = vec![0_usize; recorder.nodes.len()];
    for (_, target) in &recorder.edges {
        indegree[target.index as usize] += 1;
    }
    let mut ready = recorder
        .nodes
        .iter()
        .filter(|node| indegree[node.id.index as usize] == 0)
        .map(|node| node.id)
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(recorder.nodes.len());

    while let Some(node) = ready.pop_first() {
        order.push(node);
        for target in recorder
            .edges
            .iter()
            .filter_map(|(source, target)| (*source == node).then_some(*target))
        {
            let degree = &mut indegree[target.index as usize];
            *degree -= 1;
            if *degree == 0 {
                ready.insert(target);
            }
        }
    }

    if order.len() != recorder.nodes.len() {
        let nodes = recorder
            .nodes
            .iter()
            .filter(|node| indegree[node.id.index as usize] != 0)
            .map(|node| node.id)
            .collect();
        return Err(CompileError::Cycle(nodes));
    }
    Ok(order)
}

fn reachability(recorder: &Recorder, order: &[NodeId]) -> Vec<Vec<bool>> {
    let count = recorder.nodes.len();
    let mut reachable = vec![vec![false; count]; count];
    for node in order.iter().rev() {
        for target in recorder
            .edges
            .iter()
            .filter_map(|(source, target)| (*source == *node).then_some(*target))
        {
            reachable[node.index as usize][target.index as usize] = true;
            let target_descendants = reachable[target.index as usize].clone();
            for (is_reachable, target_reachable) in reachable[node.index as usize]
                .iter_mut()
                .zip(target_descendants)
            {
                *is_reachable |= target_reachable;
            }
        }
    }
    reachable
}

fn unordered_hazards(recorder: &Recorder, reachable: &[Vec<bool>]) -> Vec<Hazard> {
    let mut hazards = Vec::new();
    for (first_index, first_reachable) in reachable.iter().enumerate() {
        for (second_index, second_reachable) in reachable.iter().enumerate().skip(first_index + 1) {
            if first_reachable[second_index] || second_reachable[first_index] {
                continue;
            }
            let first = &recorder.nodes[first_index];
            let second = &recorder.nodes[second_index];
            for first_access in &first.accesses {
                for second_access in &second.accesses {
                    let Some(overlap) = first_access.region().intersection(second_access.region())
                    else {
                        continue;
                    };
                    let kind = match (first_access.mode(), second_access.mode()) {
                        (AccessMode::Read, AccessMode::Read) => continue,
                        (AccessMode::Write, AccessMode::Read) => HazardKind::ReadAfterWrite,
                        (AccessMode::Read, AccessMode::Write) => HazardKind::WriteAfterRead,
                        (AccessMode::Write, AccessMode::Write) => HazardKind::WriteAfterWrite,
                    };
                    hazards.push(Hazard {
                        first: first.id,
                        second: second.id,
                        kind,
                        overlap,
                    });
                }
            }
        }
    }
    hazards
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("dispatch dependency graph contains a cycle involving {0:?}")]
    Cycle(Vec<NodeId>),
    #[error("dispatch DAG has {} unordered overlapping memory hazards", .0.len())]
    UnorderedHazards(Vec<Hazard>),
    #[error("kernel {kernel:?} has an invalid kernarg ABI: {cause}")]
    InvalidKernargAbi {
        kernel: String,
        #[source]
        cause: KernargAbiError,
    },
    #[error("scalar slot {slot:?} has conflicting sizes {first} and {second}")]
    ConflictingScalarSlotSize {
        slot: ScalarSlotId,
        first: u32,
        second: u32,
    },
    #[error("kernel key {kernel:?} is used with conflicting ABI or artifact identities")]
    ConflictingKernelIdentity { kernel: String },
}

impl CompileError {
    pub fn hazards(&self) -> &[Hazard] {
        match self {
            Self::Cycle(_) => &[],
            Self::UnorderedHazards(hazards) => hazards,
            Self::InvalidKernargAbi { .. }
            | Self::ConflictingScalarSlotSize { .. }
            | Self::ConflictingKernelIdentity { .. } => &[],
        }
    }
}
