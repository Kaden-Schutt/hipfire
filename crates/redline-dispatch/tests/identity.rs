use std::ffi::c_void;
use std::num::NonZeroUsize;

use redline_dispatch::{
    Access, AllocationPolicy, ArtifactCatalog, BindingRevision, CompileError, CompileOptions, Dim3,
    KernargAbi, KernargAbiError, KernargField, KernelArg, KernelArtifactIdentity, KernelLaunch,
    PreparedPlanInvalidation, PreparedPlanStamp, PreparedPlanState, Recorder, ReplayBindingError,
    ReplayBindings, ReplayMode, ResourceBinding, ScalarSlotId,
};

fn options() -> CompileOptions {
    CompileOptions::new(NonZeroUsize::new(1).unwrap(), ReplayMode::TokenLatency)
}

fn artifact(generation: u64) -> KernelArtifactIdentity {
    KernelArtifactIdentity::from_bytes(b"code-object", b"symbol-text", generation)
}

fn abi() -> KernargAbi {
    KernargAbi::new(
        16,
        8,
        [
            KernargField::new(0, 4, 4).unwrap(),
            KernargField::new(8, 8, 8).unwrap(),
        ],
    )
    .unwrap()
}

fn dynamic_plan() -> (
    redline_dispatch::CompiledPlan,
    redline_dispatch::ResourceId,
    ScalarSlotId,
) {
    let mut recorder = Recorder::new();
    let resource = recorder.resource("buffer", 64).unwrap();
    let whole = recorder.region(resource, 0, 64).unwrap();
    let slot = ScalarSlotId::new(7);
    let launch = KernelLaunch::new("kernel", Dim3::x(1).unwrap(), Dim3::x(64).unwrap())
        .unwrap()
        .with_arguments([
            KernelArg::scalar_slot(slot, 4).unwrap(),
            KernelArg::resource(resource, 0),
        ])
        .with_kernarg_abi(abi())
        .with_artifact_identity(artifact(1));
    recorder.dispatch(launch, [Access::read(whole)]).unwrap();
    (recorder.compile(options()).unwrap(), resource, slot)
}

fn bindings(
    resource: redline_dispatch::ResourceId,
    slot: ScalarSlotId,
    address: usize,
    revision: u64,
    policy: AllocationPolicy,
    scalar: [u8; 4],
) -> ReplayBindings {
    let mut bindings = ReplayBindings::new();
    // SAFETY: tests never submit or dereference the sentinel range. It is used
    // only by pure address/layout validation.
    let resource_binding = unsafe {
        ResourceBinding::new(
            address as *mut c_void,
            64,
            BindingRevision(revision),
            policy,
        )
        .unwrap()
    };
    bindings.bind_resource(resource, resource_binding);
    bindings.bind_scalar(slot, scalar);
    bindings
}

#[test]
fn kernarg_layout_rejects_misalignment_overlap_and_bounds() {
    assert!(matches!(
        KernargField::new(2, 4, 4),
        Err(KernargAbiError::MisalignedField { .. })
    ));
    let overlap = KernargAbi::new(
        16,
        8,
        [
            KernargField::new(0, 8, 8).unwrap(),
            KernargField::new(4, 4, 4).unwrap(),
        ],
    );
    assert!(matches!(
        overlap,
        Err(KernargAbiError::OverlappingFields { .. })
    ));
    let out_of_bounds = KernargAbi::new(8, 8, [KernargField::new(8, 4, 4).unwrap()]);
    assert!(matches!(
        out_of_bounds,
        Err(KernargAbiError::FieldOutOfBounds { .. })
    ));
}

#[test]
fn kernarg_hash_is_deterministic_and_layout_sensitive() {
    assert_eq!(abi().hash(), abi().hash());
    let changed = KernargAbi::new(
        24,
        8,
        [
            KernargField::new(0, 4, 4).unwrap(),
            KernargField::new(16, 8, 8).unwrap(),
        ],
    )
    .unwrap();
    assert_ne!(abi().hash(), changed.hash());
}

#[test]
fn compile_checks_argument_layout_and_scalar_slot_consistency() {
    let mut recorder = Recorder::new();
    let slot = ScalarSlotId::new(1);
    let invalid = KernelLaunch::new("invalid", Dim3::x(1).unwrap(), Dim3::x(1).unwrap())
        .unwrap()
        .with_arguments([KernelArg::scalar_slot(slot, 8).unwrap()])
        .with_kernarg_abi(KernargAbi::new(4, 4, [KernargField::new(0, 4, 4).unwrap()]).unwrap());
    recorder.dispatch(invalid, []).unwrap();
    assert!(matches!(
        recorder.compile(options()),
        Err(CompileError::InvalidKernargAbi { .. })
    ));

    let mut recorder = Recorder::new();
    recorder
        .dispatch(
            KernelLaunch::new("a", Dim3::x(1).unwrap(), Dim3::x(1).unwrap())
                .unwrap()
                .with_arguments([KernelArg::scalar_slot(slot, 4).unwrap()]),
            [],
        )
        .unwrap();
    recorder
        .dispatch(
            KernelLaunch::new("b", Dim3::x(1).unwrap(), Dim3::x(1).unwrap())
                .unwrap()
                .with_arguments([KernelArg::scalar_slot(slot, 8).unwrap()]),
            [],
        )
        .unwrap();
    assert!(matches!(
        recorder.compile(options()),
        Err(CompileError::ConflictingScalarSlotSize { .. })
    ));
}

#[test]
fn compile_rejects_conflicting_artifact_generations_for_one_kernel_key() {
    let mut recorder = Recorder::new();
    for generation in [1, 2] {
        recorder
            .dispatch(
                KernelLaunch::new("same", Dim3::x(1).unwrap(), Dim3::x(1).unwrap())
                    .unwrap()
                    .with_artifact_identity(artifact(generation)),
                [],
            )
            .unwrap();
    }
    assert!(matches!(
        recorder.compile(options()),
        Err(CompileError::ConflictingKernelIdentity { .. })
    ));
}

#[test]
fn plan_fingerprint_normalizes_recorder_owners_and_excludes_bindings() {
    let (first, first_resource, first_slot) = dynamic_plan();
    let (second, second_resource, second_slot) = dynamic_plan();
    assert_eq!(first.fingerprint(), second.fingerprint());

    let first_bindings = bindings(
        first_resource,
        first_slot,
        0x1000,
        1,
        AllocationPolicy::HipCoarse,
        [1, 2, 3, 4],
    );
    let second_bindings = bindings(
        second_resource,
        second_slot,
        0x2000,
        99,
        AllocationPolicy::HipCoarse,
        [9, 8, 7, 6],
    );
    assert_eq!(
        first_bindings.layout_fingerprint(&first).unwrap(),
        second_bindings.layout_fingerprint(&second).unwrap()
    );
}

#[test]
fn plan_fingerprint_changes_for_shape_abi_artifact_and_topology() {
    let (base, _, _) = dynamic_plan();
    let changed_abi = KernargAbi::new(
        24,
        8,
        [
            KernargField::new(0, 4, 4).unwrap(),
            KernargField::new(16, 8, 8).unwrap(),
        ],
    )
    .unwrap();
    let make = |grid: u32, layout: KernargAbi, generation: u64, add_child: bool| {
        let mut recorder = Recorder::new();
        let resource = recorder.resource("buffer", 64).unwrap();
        let whole = recorder.region(resource, 0, 64).unwrap();
        let slot = ScalarSlotId::new(7);
        let launch = KernelLaunch::new("kernel", Dim3::x(grid).unwrap(), Dim3::x(64).unwrap())
            .unwrap()
            .with_arguments([
                KernelArg::scalar_slot(slot, 4).unwrap(),
                KernelArg::resource(resource, 0),
            ])
            .with_kernarg_abi(layout)
            .with_artifact_identity(artifact(generation));
        let root = recorder.dispatch(launch, [Access::read(whole)]).unwrap();
        if add_child {
            let child = recorder
                .dispatch(
                    KernelLaunch::new("child", Dim3::x(1).unwrap(), Dim3::x(1).unwrap()).unwrap(),
                    [],
                )
                .unwrap();
            recorder.depends_on(child, root).unwrap();
        }
        recorder.compile(options()).unwrap()
    };

    let shape = make(2, abi(), 1, false);
    let layout = make(1, changed_abi, 1, false);
    let artifact = make(1, abi(), 2, false);
    let topology = make(1, abi(), 1, true);
    for changed in [shape, layout, artifact, topology] {
        assert_ne!(base.fingerprint(), changed.fingerprint());
    }
}

#[test]
fn replay_bindings_require_exact_scalar_bytes_and_nonaliasing_resources() {
    let (plan, resource, slot) = dynamic_plan();
    let mut missing_scalar = ReplayBindings::new();
    // SAFETY: pure validation only; never submitted or dereferenced.
    missing_scalar.bind_resource(resource, unsafe {
        ResourceBinding::new(
            0x1000usize as *mut c_void,
            64,
            BindingRevision(1),
            AllocationPolicy::HipCoarse,
        )
        .unwrap()
    });
    assert!(matches!(
        missing_scalar.validate(&plan),
        Err(ReplayBindingError::ScalarNotBound { .. })
    ));
    missing_scalar.bind_scalar(slot, [1_u8, 2, 3]);
    assert!(matches!(
        missing_scalar.validate(&plan),
        Err(ReplayBindingError::ScalarSize { .. })
    ));

    let mut recorder = Recorder::new();
    let left = recorder.resource("left", 64).unwrap();
    let right = recorder.resource("right", 64).unwrap();
    recorder
        .dispatch(
            KernelLaunch::new("noop", Dim3::x(1).unwrap(), Dim3::x(1).unwrap()).unwrap(),
            [],
        )
        .unwrap();
    let plan = recorder.compile(options()).unwrap();
    let mut bindings = ReplayBindings::new();
    for resource in [left, right] {
        bindings.bind_resource(
            resource,
            // SAFETY: pure validation only; never submitted or dereferenced.
            unsafe {
                ResourceBinding::new(
                    0x4000usize as *mut c_void,
                    64,
                    BindingRevision(1),
                    AllocationPolicy::External,
                )
                .unwrap()
            },
        );
    }
    assert!(matches!(
        bindings.validate(&plan),
        Err(ReplayBindingError::AliasingResources { .. })
    ));
}

#[test]
fn prepared_stamp_distinguishes_scalar_updates_rebinds_policy_and_artifacts() {
    let (plan, resource, slot) = dynamic_plan();
    let initial = bindings(
        resource,
        slot,
        0x1000,
        1,
        AllocationPolicy::HipCoarse,
        [1, 2, 3, 4],
    );
    let stamp = PreparedPlanStamp::capture(&plan, &initial).unwrap();
    let mut catalog = ArtifactCatalog::new();
    catalog.insert("kernel", artifact(1)).unwrap();
    assert_eq!(
        stamp.validate(&plan, &initial, &catalog).unwrap(),
        PreparedPlanState::Current
    );

    let scalar_changed = bindings(
        resource,
        slot,
        0x1000,
        1,
        AllocationPolicy::HipCoarse,
        [9, 8, 7, 6],
    );
    assert_eq!(
        stamp.validate(&plan, &scalar_changed, &catalog).unwrap(),
        PreparedPlanState::Current
    );

    let rebound = bindings(
        resource,
        slot,
        0x2000,
        2,
        AllocationPolicy::HipCoarse,
        [9, 8, 7, 6],
    );
    assert_eq!(
        stamp.validate(&plan, &rebound, &catalog).unwrap(),
        PreparedPlanState::NeedsRebind {
            resources: vec![resource]
        }
    );

    let policy_changed = bindings(
        resource,
        slot,
        0x2000,
        2,
        AllocationPolicy::Uncached,
        [9, 8, 7, 6],
    );
    assert!(matches!(
        stamp.validate(&plan, &policy_changed, &catalog),
        Err(PreparedPlanInvalidation::BindingLayoutChanged { .. })
    ));

    catalog.insert("kernel", artifact(2)).unwrap();
    assert!(matches!(
        stamp.validate(&plan, &initial, &catalog),
        Err(PreparedPlanInvalidation::ArtifactChanged { .. })
    ));
}
