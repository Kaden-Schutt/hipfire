use crate::{AccessMode, CompiledPlan, DeviceRegion, HazardKind, LaneId, NodeId, PlannedDispatch};
use redline_rocr::{FenceScope, HeaderPolicy};

/// API ownership immediately before an AQL plan begins.
///
/// `HostToAql` and `HipToAql` require the producer to be quiescent before the
/// plan is submitted. A packet acquire fence establishes visibility; it does
/// not itself wait for asynchronous work in another API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntryBoundary {
    HostToAql,
    HipToAql,
    AqlToAql,
}

/// API ownership immediately after an AQL plan completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitBoundary {
    AqlToHost,
    AqlToHip,
    AqlToAql,
}

/// Visibility contract at the outside edges of one prepared AQL replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApiBoundary {
    pub entry: EntryBoundary,
    pub exit: ExitBoundary,
}

impl ApiBoundary {
    pub const fn new(entry: EntryBoundary, exit: ExitBoundary) -> Self {
        Self { entry, exit }
    }
}

impl Default for ApiBoundary {
    fn default() -> Self {
        Self::new(EntryBoundary::HostToAql, ExitBoundary::AqlToHost)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VisibilityReason {
    Entry(EntryBoundary),
    RawProducer {
        consumer: NodeId,
        overlap: DeviceRegion,
    },
    RawConsumer {
        producer: NodeId,
        overlap: DeviceRegion,
    },
    Exit(ExitBoundary),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrderReason {
    OrderedAfter(NodeId),
    CrossLaneDependency(NodeId),
    MemoryConflict {
        with: NodeId,
        kind: HazardKind,
        overlap: DeviceRegion,
    },
}

/// Why a dispatch does or does not wait for preceding packets in its queue.
///
/// An elision proof covers the entire open queue epoch, not only the adjacent
/// predecessor. A barriered dispatch can itself still be active when a later
/// unbarriered packet launches, so adjacency alone is not a sufficient proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BarrierDecision {
    Required(Vec<OrderReason>),
    Elided {
        adjacent_predecessor: Option<NodeId>,
        independent_of_open_epoch: Vec<NodeId>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedDispatchPolicy {
    node: NodeId,
    lane: LaneId,
    header: HeaderPolicy,
    barrier: BarrierDecision,
    acquire_reasons: Vec<VisibilityReason>,
    release_reasons: Vec<VisibilityReason>,
}

impl DerivedDispatchPolicy {
    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn lane(&self) -> LaneId {
        self.lane
    }

    pub fn header(&self) -> HeaderPolicy {
        self.header
    }

    pub fn barrier(&self) -> &BarrierDecision {
        &self.barrier
    }

    pub fn acquire_reasons(&self) -> &[VisibilityReason] {
        &self.acquire_reasons
    }

    pub fn release_reasons(&self) -> &[VisibilityReason] {
        &self.release_reasons
    }
}

/// Queue-local completion packet required before cross-queue fan-in.
///
/// A completion signal on the last unbarriered dispatch does not prove that
/// earlier independent dispatches in the same queue have completed. Each
/// nonempty lane therefore ends in a barriered, fence-free consolidation
/// packet whose signal is the only signal consumed by terminal fan-in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaneConsolidationPolicy {
    lane: LaneId,
    header: HeaderPolicy,
    covered_nodes: Vec<NodeId>,
}

impl LaneConsolidationPolicy {
    pub fn lane(&self) -> LaneId {
        self.lane
    }

    pub fn header(&self) -> HeaderPolicy {
        self.header
    }

    pub fn covered_nodes(&self) -> &[NodeId] {
        &self.covered_nodes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TerminalVisibilityPolicy {
    header: HeaderPolicy,
    lanes: Vec<LaneId>,
    reason: VisibilityReason,
}

impl TerminalVisibilityPolicy {
    pub fn header(&self) -> HeaderPolicy {
        self.header
    }

    pub fn lanes(&self) -> &[LaneId] {
        &self.lanes
    }

    pub fn reason(&self) -> &VisibilityReason {
        &self.reason
    }
}

/// Inspectable packet-ordering and visibility policy derived from a validated
/// backend-neutral dispatch plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedVisibilityPlan {
    boundary: ApiBoundary,
    dispatches: Vec<DerivedDispatchPolicy>,
    lane_consolidations: Vec<LaneConsolidationPolicy>,
    terminal: TerminalVisibilityPolicy,
}

impl DerivedVisibilityPlan {
    pub fn boundary(&self) -> ApiBoundary {
        self.boundary
    }

    pub fn dispatches(&self) -> &[DerivedDispatchPolicy] {
        &self.dispatches
    }

    pub fn lane_consolidations(&self) -> &[LaneConsolidationPolicy] {
        &self.lane_consolidations
    }

    pub fn terminal(&self) -> &TerminalVisibilityPolicy {
        &self.terminal
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum VisibilityError {
    #[error("cannot derive AQL visibility for an empty plan")]
    EmptyPlan,
    #[error(
        "validated plan unexpectedly contains unordered conflict {first:?} -> {second:?} ({kind:?})"
    )]
    UnorderedConflict {
        first: NodeId,
        second: NodeId,
        kind: HazardKind,
        overlap: DeviceRegion,
    },
}

/// Derive cache visibility and queue-ordering headers from plan hazards.
///
/// Distinct logical resources are assumed not to alias, as required by
/// [`crate::DeviceRegion`]. A concrete AQL preparer must validate that its
/// physical bindings preserve that invariant before using any barrier elision
/// returned here.
pub fn derive_aql_visibility(
    plan: &CompiledPlan,
    boundary: ApiBoundary,
) -> Result<DerivedVisibilityPlan, VisibilityError> {
    if plan.dispatches().is_empty() {
        return Err(VisibilityError::EmptyPlan);
    }

    let dispatches = plan.dispatches();
    let reachable = reachability(dispatches);
    validate_ordered_conflicts(dispatches, &reachable)?;

    let mut acquire_reasons = vec![Vec::new(); dispatches.len()];
    let mut release_reasons = vec![Vec::new(); dispatches.len()];
    derive_raw_visibility(
        dispatches,
        &reachable,
        &mut acquire_reasons,
        &mut release_reasons,
    );

    let lane_count = plan.lane_count().get();
    let mut lane_epochs = vec![Vec::<usize>::new(); lane_count];
    let mut lane_nodes = vec![Vec::<NodeId>::new(); lane_count];
    let mut lane_previous = vec![None::<NodeId>; lane_count];
    let mut lane_by_node = vec![LaneId(0); dispatches.len()];
    for dispatch in dispatches {
        lane_by_node[dispatch.node().index() as usize] = dispatch.lane();
    }
    let mut derived = Vec::with_capacity(dispatches.len());

    for (dispatch_index, dispatch) in dispatches.iter().enumerate() {
        let lane = dispatch.lane().0;
        let epoch = &lane_epochs[lane];
        let mut order_reasons = Vec::new();

        for dependency in dispatch.dependencies() {
            let dependency_index = dependency.index() as usize;
            if lane_by_node[dependency_index] != dispatch.lane() {
                push_unique(
                    &mut order_reasons,
                    OrderReason::CrossLaneDependency(*dependency),
                );
            }
        }

        for &prior_index in epoch {
            let prior = &dispatches[prior_index];
            if reachable[prior.node().index() as usize][dispatch.node().index() as usize] {
                push_unique(&mut order_reasons, OrderReason::OrderedAfter(prior.node()));
            }
            for conflict in conflicts(prior, dispatch) {
                push_unique(
                    &mut order_reasons,
                    OrderReason::MemoryConflict {
                        with: prior.node(),
                        kind: conflict.kind,
                        overlap: conflict.overlap,
                    },
                );
            }
        }

        let (barrier, decision) = if order_reasons.is_empty() {
            (
                false,
                BarrierDecision::Elided {
                    adjacent_predecessor: lane_previous[lane],
                    independent_of_open_epoch: epoch
                        .iter()
                        .map(|index| dispatches[*index].node())
                        .collect(),
                },
            )
        } else {
            (true, BarrierDecision::Required(order_reasons))
        };

        let mut acquire = if acquire_reasons[dispatch_index].is_empty() {
            FenceScope::None
        } else {
            FenceScope::Agent
        };
        if lane_nodes[lane].is_empty() {
            acquire = max_scope(acquire, entry_scope(boundary.entry));
            acquire_reasons[dispatch_index].insert(0, VisibilityReason::Entry(boundary.entry));
        }
        let release = if release_reasons[dispatch_index].is_empty() {
            FenceScope::None
        } else {
            FenceScope::Agent
        };

        derived.push(DerivedDispatchPolicy {
            node: dispatch.node(),
            lane: dispatch.lane(),
            header: HeaderPolicy {
                barrier,
                acquire,
                release,
            },
            barrier: decision,
            acquire_reasons: std::mem::take(&mut acquire_reasons[dispatch_index]),
            release_reasons: std::mem::take(&mut release_reasons[dispatch_index]),
        });

        if barrier {
            lane_epochs[lane].clear();
        }
        lane_epochs[lane].push(dispatch_index);
        lane_nodes[lane].push(dispatch.node());
        lane_previous[lane] = Some(dispatch.node());
    }

    let consolidation_header = HeaderPolicy {
        barrier: true,
        acquire: FenceScope::None,
        release: FenceScope::None,
    };
    let lane_consolidations = lane_nodes
        .into_iter()
        .enumerate()
        .filter_map(|(lane, covered_nodes)| {
            (!covered_nodes.is_empty()).then_some(LaneConsolidationPolicy {
                lane: LaneId(lane),
                header: consolidation_header,
                covered_nodes,
            })
        })
        .collect::<Vec<_>>();
    let terminal_lanes = lane_consolidations
        .iter()
        .map(LaneConsolidationPolicy::lane)
        .collect();
    let terminal = TerminalVisibilityPolicy {
        header: HeaderPolicy {
            barrier: true,
            acquire: FenceScope::None,
            release: exit_scope(boundary.exit),
        },
        lanes: terminal_lanes,
        reason: VisibilityReason::Exit(boundary.exit),
    };

    Ok(DerivedVisibilityPlan {
        boundary,
        dispatches: derived,
        lane_consolidations,
        terminal,
    })
}

fn entry_scope(boundary: EntryBoundary) -> FenceScope {
    match boundary {
        EntryBoundary::HostToAql | EntryBoundary::HipToAql => FenceScope::System,
        EntryBoundary::AqlToAql => FenceScope::Agent,
    }
}

fn exit_scope(boundary: ExitBoundary) -> FenceScope {
    match boundary {
        ExitBoundary::AqlToHost | ExitBoundary::AqlToHip => FenceScope::System,
        ExitBoundary::AqlToAql => FenceScope::Agent,
    }
}

fn max_scope(left: FenceScope, right: FenceScope) -> FenceScope {
    if (left as u16) >= (right as u16) {
        left
    } else {
        right
    }
}

fn reachability(dispatches: &[PlannedDispatch]) -> Vec<Vec<bool>> {
    let count = dispatches.len();
    let mut reachable = vec![vec![false; count]; count];
    for dispatch in dispatches.iter().rev() {
        let node = dispatch.node().index() as usize;
        for dependency in dispatch.dependencies() {
            let dependency = dependency.index() as usize;
            reachable[dependency][node] = true;
            let descendants = reachable[node].clone();
            for (known, descendant) in reachable[dependency].iter_mut().zip(descendants) {
                *known |= descendant;
            }
        }
    }
    reachable
}

#[derive(Clone, Copy)]
struct Conflict {
    kind: HazardKind,
    overlap: DeviceRegion,
}

fn conflicts(first: &PlannedDispatch, second: &PlannedDispatch) -> Vec<Conflict> {
    let mut result = Vec::new();
    for first_access in first.accesses() {
        for second_access in second.accesses() {
            let Some(overlap) = first_access.region().intersection(second_access.region()) else {
                continue;
            };
            let kind = match (first_access.mode(), second_access.mode()) {
                (AccessMode::Read, AccessMode::Read) => continue,
                (AccessMode::Write, AccessMode::Read) => HazardKind::ReadAfterWrite,
                (AccessMode::Read, AccessMode::Write) => HazardKind::WriteAfterRead,
                (AccessMode::Write, AccessMode::Write) => HazardKind::WriteAfterWrite,
            };
            result.push(Conflict { kind, overlap });
        }
    }
    result
}

fn validate_ordered_conflicts(
    dispatches: &[PlannedDispatch],
    reachable: &[Vec<bool>],
) -> Result<(), VisibilityError> {
    for first in 0..dispatches.len() {
        for second in first + 1..dispatches.len() {
            for conflict in conflicts(&dispatches[first], &dispatches[second]) {
                if !reachable[dispatches[first].node().index() as usize]
                    [dispatches[second].node().index() as usize]
                {
                    return Err(VisibilityError::UnorderedConflict {
                        first: dispatches[first].node(),
                        second: dispatches[second].node(),
                        kind: conflict.kind,
                        overlap: conflict.overlap,
                    });
                }
            }
        }
    }
    Ok(())
}

fn derive_raw_visibility(
    dispatches: &[PlannedDispatch],
    reachable: &[Vec<bool>],
    acquire_reasons: &mut [Vec<VisibilityReason>],
    release_reasons: &mut [Vec<VisibilityReason>],
) {
    for first in 0..dispatches.len() {
        for second in first + 1..dispatches.len() {
            if !reachable[dispatches[first].node().index() as usize]
                [dispatches[second].node().index() as usize]
            {
                continue;
            }
            for conflict in conflicts(&dispatches[first], &dispatches[second]) {
                if conflict.kind != HazardKind::ReadAfterWrite {
                    continue;
                }
                release_reasons[first].push(VisibilityReason::RawProducer {
                    consumer: dispatches[second].node(),
                    overlap: conflict.overlap,
                });
                acquire_reasons[second].push(VisibilityReason::RawConsumer {
                    producer: dispatches[first].node(),
                    overlap: conflict.overlap,
                });
            }
        }
    }
}

fn push_unique<T: Eq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Access, CompileOptions, Dim3, KernelLaunch, Recorder, ReplayMode};
    use std::num::{NonZeroU32, NonZeroUsize};

    fn launch(name: &str, work: u32) -> KernelLaunch {
        KernelLaunch::new(name, Dim3::x(1).unwrap(), Dim3::x(32).unwrap())
            .unwrap()
            .with_estimated_work(NonZeroU32::new(work).unwrap())
    }

    fn compile(recorder: &Recorder, lanes: usize) -> CompiledPlan {
        recorder
            .compile(CompileOptions::new(
                NonZeroUsize::new(lanes).unwrap(),
                ReplayMode::TokenLatency,
            ))
            .unwrap()
    }

    fn encoded_header(policy: HeaderPolicy, packet_type: u16) -> u16 {
        packet_type
            | (u16::from(policy.barrier) << 8)
            | ((policy.acquire as u16) << 9)
            | ((policy.release as u16) << 11)
    }

    fn dispatch_header(policy: HeaderPolicy) -> u16 {
        encoded_header(policy, 2)
    }

    fn barrier_header(policy: HeaderPolicy) -> u16 {
        encoded_header(policy, 3)
    }

    #[test]
    fn step_one_header_encodings_remain_frozen() {
        assert_eq!(dispatch_header(HeaderPolicy::RECORDED_DISPATCH), 0x1502);
        assert_eq!(dispatch_header(HeaderPolicy::TWO_QUEUE_DISPATCH), 0x0d02);
        assert_eq!(
            dispatch_header(HeaderPolicy::BATCH_BOUNDARY_FIRST_SERIAL),
            0x0502
        );
        assert_eq!(
            dispatch_header(HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL),
            0x0102
        );
        assert_eq!(
            dispatch_header(HeaderPolicy::BATCH_BOUNDARY_FIRST_INDEPENDENT),
            0x0402
        );
        assert_eq!(
            dispatch_header(HeaderPolicy::BATCH_BOUNDARY_INTERNAL_INDEPENDENT),
            0x0002
        );
        assert_eq!(barrier_header(HeaderPolicy::TWO_QUEUE_DEPENDENCY), 0x0103);
        assert_eq!(
            barrier_header(HeaderPolicy::TWO_QUEUE_HOST_TERMINAL),
            0x1103
        );
    }

    #[test]
    fn independent_batch_elides_only_with_full_epoch_proof_and_consolidates() {
        let mut recorder = Recorder::new();
        let weights = recorder.resource("weights", 64).unwrap();
        let output_resource = recorder.resource("output", 192).unwrap();
        let weights = recorder.region(weights, 0, 64).unwrap();
        for index in 0..3 {
            let output = recorder.region(output_resource, index * 64, 64).unwrap();
            recorder
                .dispatch(
                    launch(&format!("gemv-{index}"), 1),
                    [Access::read(weights), Access::write(output)],
                )
                .unwrap();
        }

        let plan = derive_aql_visibility(&compile(&recorder, 1), ApiBoundary::default()).unwrap();
        assert_eq!(
            plan.dispatches()
                .iter()
                .map(|policy| dispatch_header(policy.header()))
                .collect::<Vec<_>>(),
            vec![0x0402, 0x0002, 0x0002]
        );
        assert!(matches!(
            plan.dispatches()[2].barrier(),
            BarrierDecision::Elided {
                independent_of_open_epoch,
                ..
            } if independent_of_open_epoch.len() == 2
        ));
        assert_eq!(plan.lane_consolidations().len(), 1);
        assert_eq!(
            barrier_header(plan.lane_consolidations()[0].header()),
            0x0103
        );
        assert_eq!(barrier_header(plan.terminal().header()), 0x1103);
    }

    #[test]
    fn non_adjacent_dependency_forces_barrier_and_resets_epoch() {
        let mut recorder = Recorder::new();
        let a = recorder.dispatch(launch("a", 1), []).unwrap();
        recorder.dispatch(launch("b", 1), []).unwrap();
        let c = recorder.dispatch(launch("c", 1), []).unwrap();
        let d = recorder.dispatch(launch("d", 1), []).unwrap();
        recorder.depends_on(c, a).unwrap();

        let plan = derive_aql_visibility(&compile(&recorder, 1), ApiBoundary::default()).unwrap();
        assert!(!plan.dispatches()[0].header().barrier);
        assert!(!plan.dispatches()[1].header().barrier);
        assert!(plan.dispatches()[2].header().barrier);
        assert!(matches!(
            plan.dispatches()[2].barrier(),
            BarrierDecision::Required(reasons)
                if reasons.contains(&OrderReason::OrderedAfter(a))
        ));
        assert!(!plan.dispatches()[3].header().barrier);
        assert!(matches!(
            plan.dispatches()[3].barrier(),
            BarrierDecision::Elided {
                independent_of_open_epoch,
                ..
            } if independent_of_open_epoch == &[c]
        ));
        assert_eq!(plan.dispatches()[3].node(), d);
    }

    #[test]
    fn cross_lane_raw_gets_agent_visibility_and_queue_consolidation() {
        let mut recorder = Recorder::new();
        let transfer = recorder.resource("transfer", 64).unwrap();
        let transfer = recorder.region(transfer, 0, 64).unwrap();
        recorder
            .dispatch(launch("lane-zero-anchor", 10), [])
            .unwrap();
        let producer = recorder
            .dispatch(launch("producer", 1), [Access::write(transfer)])
            .unwrap();
        recorder
            .dispatch(launch("lane-one-blocker", 20), [])
            .unwrap();
        let consumer = recorder
            .dispatch(launch("consumer", 1), [Access::read(transfer)])
            .unwrap();
        recorder.depends_on(consumer, producer).unwrap();

        let compiled = compile(&recorder, 2);
        assert_ne!(
            compiled.dispatches()[producer.index() as usize].lane(),
            compiled.dispatches()[consumer.index() as usize].lane()
        );
        let plan = derive_aql_visibility(&compiled, ApiBoundary::default()).unwrap();
        let producer = &plan.dispatches()[producer.index() as usize];
        let consumer = &plan.dispatches()[consumer.index() as usize];
        assert_eq!(producer.header().release, FenceScope::Agent);
        assert_eq!(consumer.header().acquire, FenceScope::Agent);
        assert!(consumer.header().barrier);
        assert!(matches!(
            consumer.barrier(),
            BarrierDecision::Required(reasons)
                if reasons.iter().any(|reason| matches!(
                    reason,
                    OrderReason::CrossLaneDependency(node) if *node == producer.node()
                ))
        ));
        assert_eq!(plan.lane_consolidations().len(), 2);
        assert_eq!(plan.terminal().lanes().len(), 2);
    }

    #[test]
    fn transitive_raw_carries_agent_release_and_acquire() {
        let mut recorder = Recorder::new();
        let transfer = recorder.resource("transfer", 64).unwrap();
        let transfer = recorder.region(transfer, 0, 64).unwrap();
        let writer = recorder
            .dispatch(launch("writer", 1), [Access::write(transfer)])
            .unwrap();
        let middle = recorder.dispatch(launch("middle", 1), []).unwrap();
        let reader = recorder
            .dispatch(launch("reader", 1), [Access::read(transfer)])
            .unwrap();
        recorder.depends_on(middle, writer).unwrap();
        recorder.depends_on(reader, middle).unwrap();

        let plan = derive_aql_visibility(&compile(&recorder, 1), ApiBoundary::default()).unwrap();
        assert_eq!(
            plan.dispatches()[writer.index() as usize].header().release,
            FenceScope::Agent
        );
        assert_eq!(
            plan.dispatches()[reader.index() as usize].header().acquire,
            FenceScope::Agent
        );
    }

    #[test]
    fn node_to_lane_lookup_survives_reordered_topology() {
        let mut recorder = Recorder::new();
        let late = recorder.dispatch(launch("late-node-zero", 1), []).unwrap();
        recorder
            .dispatch(launch("lane-zero-anchor", 10), [])
            .unwrap();
        let first_dependency = recorder
            .dispatch(launch("first-dependency", 1), [])
            .unwrap();
        let second_dependency = recorder
            .dispatch(launch("second-dependency", 20), [])
            .unwrap();
        recorder.depends_on(late, first_dependency).unwrap();
        recorder.depends_on(late, second_dependency).unwrap();

        let compiled = compile(&recorder, 2);
        assert_ne!(compiled.dispatches()[0].node(), late);
        let plan = derive_aql_visibility(&compiled, ApiBoundary::default()).unwrap();
        let late_policy = plan
            .dispatches()
            .iter()
            .find(|policy| policy.node() == late)
            .unwrap();
        assert!(late_policy.header().barrier);
        assert!(matches!(
            late_policy.barrier(),
            BarrierDecision::Required(reasons)
                if reasons.iter().any(|reason| matches!(
                    reason,
                    OrderReason::CrossLaneDependency(_)
                ))
        ));
    }

    #[test]
    fn war_and_waw_order_without_claiming_raw_visibility() {
        for (first, second) in [
            (AccessMode::Read, AccessMode::Write),
            (AccessMode::Write, AccessMode::Write),
        ] {
            let mut recorder = Recorder::new();
            let resource = recorder.resource("resource", 64).unwrap();
            let region = recorder.region(resource, 0, 64).unwrap();
            let access = |mode| match mode {
                AccessMode::Read => Access::read(region),
                AccessMode::Write => Access::write(region),
            };
            let predecessor = recorder
                .dispatch(launch("predecessor", 1), [access(first)])
                .unwrap();
            let successor = recorder
                .dispatch(launch("successor", 1), [access(second)])
                .unwrap();
            recorder.depends_on(successor, predecessor).unwrap();

            let plan =
                derive_aql_visibility(&compile(&recorder, 1), ApiBoundary::default()).unwrap();
            let predecessor = &plan.dispatches()[predecessor.index() as usize];
            let successor = &plan.dispatches()[successor.index() as usize];
            assert_eq!(predecessor.header().release, FenceScope::None);
            assert_eq!(successor.header().acquire, FenceScope::None);
            assert!(successor.header().barrier);
        }
    }

    #[test]
    fn api_boundaries_map_to_system_or_agent_scopes() {
        for (entry, expected) in [
            (EntryBoundary::HostToAql, FenceScope::System),
            (EntryBoundary::HipToAql, FenceScope::System),
            (EntryBoundary::AqlToAql, FenceScope::Agent),
        ] {
            let mut recorder = Recorder::new();
            recorder.dispatch(launch("only", 1), []).unwrap();
            let compiled = compile(&recorder, 1);
            for (exit, expected_exit) in [
                (ExitBoundary::AqlToHost, FenceScope::System),
                (ExitBoundary::AqlToHip, FenceScope::System),
                (ExitBoundary::AqlToAql, FenceScope::Agent),
            ] {
                let plan = derive_aql_visibility(&compiled, ApiBoundary::new(entry, exit)).unwrap();
                assert_eq!(plan.dispatches()[0].header().acquire, expected);
                assert_eq!(plan.terminal().header().release, expected_exit);
            }
        }
    }
}
