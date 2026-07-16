//! Backend-neutral record/replay core for Redline Objective 2.
//!
//! The core records a dispatch DAG, validates memory ordering, assigns
//! dispatches to deterministic logical lanes, and drives an explicit
//! [`DispatchBackend`] implementation. [`hip`] provides the first real runtime
//! backend; [`aql`] contains the direct public-ROCr packet replay prototype.
//!
//! ```
//! use redline_dispatch::mock::MockBackend;
//! use redline_dispatch::{
//!     Access, CompileOptions, Dim3, KernelLaunch, Recorder, ReplayMode, ReplayToken,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut recorder = Recorder::new();
//! let activations = recorder.resource("activations", 4096)?;
//! let input = recorder.region(activations, 0, 2048)?;
//! let output = recorder.region(activations, 2048, 2048)?;
//! let project = KernelLaunch::new("project", Dim3::x(32)?, Dim3::x(64)?)?;
//! let project = recorder.dispatch(
//!     project,
//!     [Access::read(input), Access::write(output)],
//! )?;
//! let consume = KernelLaunch::new("consume", Dim3::x(32)?, Dim3::x(64)?)?;
//! let consume = recorder.dispatch(consume, [Access::read(output)])?;
//! recorder.depends_on(consume, project)?;
//!
//! let options = CompileOptions::lanes(4, ReplayMode::TokenLatency).unwrap();
//! let plan = recorder.compile(options)?;
//! let mut backend = MockBackend::default();
//! plan.replay(&mut backend, ReplayToken(0))?;
//! # Ok(())
//! # }
//! ```

pub mod aql;
mod backend;
mod bindings;
pub mod hip;
mod identity;
mod ir;
pub mod mock;
mod plan;
mod recorder;
mod selection;
mod visibility;

pub use backend::{
    BeginReplay, DispatchBackend, DispatchRequest, EndReplay, ReplayMode, ReplayToken,
};
pub use bindings::{
    AllocationPolicy, BindingLayoutFingerprint, BindingRevision, PreparedPlanInvalidation,
    PreparedPlanStamp, PreparedPlanState, ReplayBindingError, ReplayBindings, ResourceBinding,
};
pub use identity::{
    ArtifactCatalog, ArtifactCatalogError, KernargAbi, KernargAbiError, KernargAbiHash,
    KernargField, KernelArtifactIdentity, Sha256Digest,
};
pub use ir::{
    Access, AccessMode, DeviceRegion, Dim3, KernelArg, KernelLaunch, NodeId, ResourceId,
    ScalarSlotId,
};
pub use plan::{
    CompileError, CompileOptions, CompiledPlan, Hazard, HazardKind, LaneId, PlanFingerprint,
    PlannedDispatch, PlannedResource,
};
pub use recorder::{RecordError, Recorder};
pub use selection::{
    AutoDecision, AutoPolicy, BackendEvidence, BackendKind, PlanCacheError, PlanCacheKey,
    PreparedPlanCache,
};
pub use visibility::{
    ApiBoundary, BarrierDecision, DerivedDispatchPolicy, DerivedVisibilityPlan, EntryBoundary,
    ExitBoundary, LaneConsolidationPolicy, OrderReason, TerminalVisibilityPolicy, VisibilityError,
    VisibilityReason, derive_aql_visibility,
};
