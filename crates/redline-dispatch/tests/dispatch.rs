use std::num::{NonZeroU32, NonZeroUsize};

use redline_dispatch::mock::{MockBackend, MockCommand, MockSignal};
use redline_dispatch::{
    Access, CompileError, CompileOptions, DeviceRegion, Dim3, HazardKind, KernelArg, KernelLaunch,
    LaneId, RecordError, Recorder, ReplayMode, ReplayToken,
};

fn launch(name: &str) -> KernelLaunch {
    KernelLaunch::new(name, Dim3::x(1).unwrap(), Dim3::x(64).unwrap()).unwrap()
}

fn options(lanes: usize) -> CompileOptions {
    CompileOptions::new(NonZeroUsize::new(lanes).unwrap(), ReplayMode::TokenLatency)
}

#[test]
fn regions_are_bounds_checked_and_recorder_scoped() {
    let mut first = Recorder::new();
    let buffer = first.resource("buffer", 64).unwrap();
    assert!(matches!(
        first.region(buffer, 48, 32),
        Err(RecordError::RegionOutOfBounds { .. })
    ));
    assert!(matches!(
        DeviceRegion::new(buffer, u64::MAX, 2),
        Err(RecordError::RegionAddressOverflow { .. })
    ));

    let second = Recorder::new();
    assert!(matches!(
        second.region(buffer, 0, 1),
        Err(RecordError::ForeignResource(_))
    ));
}

#[test]
fn resource_kernel_arguments_are_validated_before_recording() {
    let mut recorder = Recorder::new();
    let buffer = recorder.resource("buffer", 64).unwrap();
    let launch = launch("kernel").with_arguments([KernelArg::resource(buffer, 64)]);
    assert!(matches!(
        recorder.dispatch(launch, []),
        Err(RecordError::ResourceArgumentOutOfBounds { .. })
    ));
}

#[test]
fn every_resource_argument_requires_a_covering_access_declaration() {
    let mut recorder = Recorder::new();
    let buffer = recorder.resource("buffer", 64).unwrap();
    let missing_access = launch("kernel").with_arguments([KernelArg::resource(buffer, 16)]);
    assert!(matches!(
        recorder.dispatch(missing_access, []),
        Err(RecordError::UndeclaredResourceArgument { .. })
    ));

    let covered_access = launch("kernel").with_arguments([KernelArg::resource(buffer, 16)]);
    let covered = recorder.region(buffer, 8, 16).unwrap();
    assert!(
        recorder
            .dispatch(covered_access, [Access::read(covered)])
            .is_ok()
    );
}

#[test]
fn compile_reports_raw_war_and_waw_for_unordered_overlaps() {
    type MakeAccess = fn(DeviceRegion) -> Access;
    let cases = [
        (
            Access::write as MakeAccess,
            Access::read as MakeAccess,
            HazardKind::ReadAfterWrite,
        ),
        (
            Access::read as MakeAccess,
            Access::write as MakeAccess,
            HazardKind::WriteAfterRead,
        ),
        (
            Access::write as MakeAccess,
            Access::write as MakeAccess,
            HazardKind::WriteAfterWrite,
        ),
    ];

    for (first_access, second_access, expected) in cases {
        let mut recorder = Recorder::new();
        let buffer = recorder.resource("buffer", 128).unwrap();
        let first_region = recorder.region(buffer, 16, 48).unwrap();
        let second_region = recorder.region(buffer, 32, 64).unwrap();
        recorder
            .dispatch(launch("first"), [first_access(first_region)])
            .unwrap();
        recorder
            .dispatch(launch("second"), [second_access(second_region)])
            .unwrap();

        let error = recorder.compile(options(2)).unwrap_err();
        let hazards = error.hazards();
        assert_eq!(hazards.len(), 1);
        assert_eq!(hazards[0].kind, expected);
        assert_eq!(hazards[0].overlap.offset(), 32);
        assert_eq!(hazards[0].overlap.len(), 32);
    }
}

#[test]
fn read_read_and_disjoint_writes_can_overlap() {
    let mut recorder = Recorder::new();
    let buffer = recorder.resource("buffer", 128).unwrap();
    let whole = recorder.region(buffer, 0, 128).unwrap();
    let left = recorder.region(buffer, 0, 64).unwrap();
    let right = recorder.region(buffer, 64, 64).unwrap();
    let reader_a = recorder
        .dispatch(launch("reader-a"), [Access::read(whole)])
        .unwrap();
    let reader_b = recorder
        .dispatch(launch("reader-b"), [Access::read(whole)])
        .unwrap();
    let writer_a = recorder
        .dispatch(launch("writer-a"), [Access::write(left)])
        .unwrap();
    let writer_b = recorder
        .dispatch(launch("writer-b"), [Access::write(right)])
        .unwrap();

    // Readers conflict with writers, so order that part while retaining the
    // independent reader-reader and disjoint writer-writer pairs.
    recorder.depends_on(writer_a, reader_a).unwrap();
    recorder.depends_on(writer_a, reader_b).unwrap();
    recorder.depends_on(writer_b, reader_a).unwrap();
    recorder.depends_on(writer_b, reader_b).unwrap();
    assert!(recorder.compile(options(4)).is_ok());
}

#[test]
fn transitive_dependency_orders_a_memory_hazard() {
    let mut recorder = Recorder::new();
    let buffer = recorder.resource("buffer", 64).unwrap();
    let region = recorder.region(buffer, 0, 64).unwrap();
    let writer = recorder
        .dispatch(launch("writer"), [Access::write(region)])
        .unwrap();
    let middle = recorder.dispatch(launch("middle"), []).unwrap();
    let reader = recorder
        .dispatch(launch("reader"), [Access::read(region)])
        .unwrap();
    recorder.depends_on(middle, writer).unwrap();
    recorder.depends_on(reader, middle).unwrap();

    assert!(recorder.compile(options(2)).is_ok());
}

#[test]
fn cycle_is_rejected_without_mutating_the_graph() {
    let mut recorder = Recorder::new();
    let first = recorder.dispatch(launch("first"), []).unwrap();
    let second = recorder.dispatch(launch("second"), []).unwrap();
    recorder.depends_on(second, first).unwrap();
    assert!(matches!(
        recorder.depends_on(first, second),
        Err(RecordError::DependencyCycle { .. })
    ));
    assert!(recorder.compile(options(2)).is_ok());
}

#[test]
fn list_schedule_is_deterministic_and_preserves_a_chain_lane() {
    let mut recorder = Recorder::new();
    let first = recorder.dispatch(launch("first"), []).unwrap();
    let independent = recorder.dispatch(launch("independent"), []).unwrap();
    let child = recorder
        .dispatch(
            launch("child").with_estimated_work(NonZeroU32::new(2).unwrap()),
            [],
        )
        .unwrap();
    recorder.depends_on(child, first).unwrap();

    let one = recorder.compile(options(2)).unwrap();
    let two = recorder.compile(options(2)).unwrap();
    assert_eq!(one, two);
    assert_eq!(one.dispatches()[0].lane(), LaneId(0));
    assert_eq!(one.dispatches()[1].node(), independent);
    assert_eq!(one.dispatches()[1].lane(), LaneId(1));
    assert_eq!(one.dispatches()[2].node(), child);
    assert_eq!(one.dispatches()[2].lane(), LaneId(0));
    assert_eq!(one.dispatches()[2].estimated_start(), 1);
    assert_eq!(one.dispatches()[2].estimated_end(), 3);
}

#[test]
fn mock_replay_exposes_lanes_dependencies_and_terminal_signals() {
    let mut recorder = Recorder::new();
    let root = recorder.dispatch(launch("root"), []).unwrap();
    let left = recorder.dispatch(launch("left"), []).unwrap();
    let right = recorder.dispatch(launch("right"), []).unwrap();
    recorder.depends_on(left, root).unwrap();
    recorder.depends_on(right, root).unwrap();
    let plan = recorder.compile(options(2)).unwrap();

    let mut backend = MockBackend::default();
    let completion = plan.replay(&mut backend, ReplayToken(7)).unwrap();
    assert_eq!(completion.token, ReplayToken(7));
    assert_eq!(
        completion.terminal_signals,
        vec![MockSignal(1), MockSignal(2)]
    );
    assert!(matches!(
        &backend.commands()[2],
        MockCommand::Dispatch { dependencies, .. } if dependencies == &[MockSignal(0)]
    ));
    assert!(matches!(
        backend.commands().last(),
        Some(MockCommand::End(end)) if end.mode == ReplayMode::TokenLatency
    ));
}

#[test]
fn throughput_replay_policy_reaches_the_backend() {
    let mut recorder = Recorder::new();
    recorder.dispatch(launch("kernel"), []).unwrap();
    let mode = ReplayMode::throughput(3).unwrap();
    let plan = recorder
        .compile(CompileOptions::new(NonZeroUsize::new(4).unwrap(), mode))
        .unwrap();
    let mut backend = MockBackend::default();
    let completion = plan.replay(&mut backend, ReplayToken(11)).unwrap();
    assert_eq!(plan.replay_mode(), mode);
    assert_eq!(completion.mode, mode);
    assert!(matches!(
        &backend.commands()[0],
        MockCommand::Begin {
            mode: ReplayMode::Throughput { max_tokens_in_flight },
            lane_count: 4,
            ..
        } if max_tokens_in_flight.get() == 3
    ));
}

#[test]
fn unordered_hazard_error_exposes_all_conflicts() {
    let mut recorder = Recorder::new();
    let buffer = recorder.resource("buffer", 64).unwrap();
    let region = recorder.region(buffer, 0, 64).unwrap();
    recorder
        .dispatch(launch("write-a"), [Access::write(region)])
        .unwrap();
    recorder
        .dispatch(launch("write-b"), [Access::write(region)])
        .unwrap();
    recorder
        .dispatch(launch("read"), [Access::read(region)])
        .unwrap();
    let error = recorder.compile(options(3)).unwrap_err();
    assert!(matches!(error, CompileError::UnorderedHazards(_)));
    assert_eq!(error.hazards().len(), 3);
}
