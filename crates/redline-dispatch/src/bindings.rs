use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::{
    ArtifactCatalog, CompiledPlan, KernelArtifactIdentity, PlanFingerprint, ResourceId,
    ScalarSlotId,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AllocationPolicy {
    HipCoarse,
    HsaCoarse,
    HsaFine,
    Uncached,
    External,
}

impl AllocationPolicy {
    const fn tag(self) -> u8 {
        match self {
            Self::HipCoarse => 0,
            Self::HsaCoarse => 1,
            Self::HsaFine => 2,
            Self::Uncached => 3,
            Self::External => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BindingRevision(pub u64);

#[derive(Clone, Copy, Debug)]
pub struct ResourceBinding {
    base: NonNull<c_void>,
    size: u64,
    revision: BindingRevision,
    policy: AllocationPolicy,
}

impl ResourceBinding {
    /// # Safety
    ///
    /// `base..base+size` must identify a live GPU-accessible allocation for
    /// every replay that uses this binding. The revision must change whenever
    /// the address or allocation identity changes.
    pub unsafe fn new(
        base: *mut c_void,
        size: u64,
        revision: BindingRevision,
        policy: AllocationPolicy,
    ) -> Result<Self, ReplayBindingError> {
        let base = NonNull::new(base).ok_or(ReplayBindingError::NullResourcePointer)?;
        if size == 0 {
            return Err(ReplayBindingError::EmptyResourceBinding);
        }
        let size_usize = usize::try_from(size)
            .map_err(|_| ReplayBindingError::ResourceAddressOverflow { size })?;
        (base.as_ptr() as usize)
            .checked_add(size_usize)
            .ok_or(ReplayBindingError::ResourceAddressOverflow { size })?;
        Ok(Self {
            base,
            size,
            revision,
            policy,
        })
    }

    pub const fn base(self) -> NonNull<c_void> {
        self.base
    }

    pub const fn size(self) -> u64 {
        self.size
    }

    pub const fn revision(self) -> BindingRevision {
        self.revision
    }

    pub const fn policy(self) -> AllocationPolicy {
        self.policy
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReplayBindings {
    resources: BTreeMap<ResourceId, ResourceBinding>,
    scalars: BTreeMap<ScalarSlotId, Arc<[u8]>>,
}

impl ReplayBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind_resource(&mut self, resource: ResourceId, binding: ResourceBinding) {
        self.resources.insert(resource, binding);
    }

    pub fn bind_scalar(&mut self, slot: ScalarSlotId, bytes: impl Into<Arc<[u8]>>) {
        self.scalars.insert(slot, bytes.into());
    }

    pub fn resource(&self, resource: ResourceId) -> Option<ResourceBinding> {
        self.resources.get(&resource).copied()
    }

    pub fn scalar(&self, slot: ScalarSlotId) -> Option<&[u8]> {
        self.scalars.get(&slot).map(AsRef::as_ref)
    }

    pub fn validate(&self, plan: &CompiledPlan) -> Result<(), ReplayBindingError> {
        for planned in plan.resources() {
            let binding = self.resources.get(&planned.id()).copied().ok_or(
                ReplayBindingError::ResourceNotBound {
                    resource: planned.id(),
                },
            )?;
            if binding.size < planned.size() {
                return Err(ReplayBindingError::ResourceTooSmall {
                    resource: planned.id(),
                    required: planned.size(),
                    bound: binding.size,
                });
            }
        }
        for (slot, expected) in plan.scalar_slots() {
            let bytes = self
                .scalars
                .get(&slot)
                .ok_or(ReplayBindingError::ScalarNotBound { slot })?;
            if bytes.len() != expected as usize {
                return Err(ReplayBindingError::ScalarSize {
                    slot,
                    expected,
                    actual: bytes.len(),
                });
            }
        }
        let resources = plan.resources();
        for (index, first) in resources.iter().enumerate() {
            let first_binding = self.resources[&first.id()];
            let first_start = first_binding.base.as_ptr() as usize;
            let first_end = first_start + first_binding.size as usize;
            for second in resources.iter().skip(index + 1) {
                let second_binding = self.resources[&second.id()];
                let second_start = second_binding.base.as_ptr() as usize;
                let second_end = second_start + second_binding.size as usize;
                if first_start < second_end && second_start < first_end {
                    return Err(ReplayBindingError::AliasingResources {
                        first: first.id(),
                        second: second.id(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn layout_fingerprint(
        &self,
        plan: &CompiledPlan,
    ) -> Result<BindingLayoutFingerprint, ReplayBindingError> {
        self.validate(plan)?;
        let mut hash = Sha256::new();
        hash.update(b"redline-binding-layout-v1\0");
        hash.update((plan.resources().len() as u64).to_le_bytes());
        for resource in plan.resources() {
            let binding = self.resources[&resource.id()];
            hash.update(resource.id().index().to_le_bytes());
            hash.update(resource.size().to_le_bytes());
            hash.update([binding.policy.tag()]);
        }
        for (slot, size) in plan.scalar_slots() {
            hash.update(slot.index().to_le_bytes());
            hash.update(size.to_le_bytes());
        }
        Ok(BindingLayoutFingerprint(hash.finalize().into()))
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BindingLayoutFingerprint([u8; 32]);

impl BindingLayoutFingerprint {
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn from_bytes_for_test(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl fmt::Debug for BindingLayoutFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0[..8] {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str("…")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedPlanState {
    Current,
    NeedsRebind { resources: Vec<ResourceId> },
}

#[derive(Clone, Debug)]
pub struct PreparedPlanStamp {
    plan: PlanFingerprint,
    binding_layout: BindingLayoutFingerprint,
    resource_revisions: BTreeMap<u32, BindingRevision>,
    artifacts: BTreeMap<String, KernelArtifactIdentity>,
}

impl PreparedPlanStamp {
    pub fn capture(
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
    ) -> Result<Self, ReplayBindingError> {
        let binding_layout = bindings.layout_fingerprint(plan)?;
        let resource_revisions = plan
            .resources()
            .iter()
            .map(|resource| {
                (
                    resource.id().index(),
                    bindings.resources[&resource.id()].revision,
                )
            })
            .collect();
        let artifacts = plan
            .artifact_identities()
            .into_iter()
            .map(|(kernel, identity)| (kernel.to_owned(), identity))
            .collect();
        Ok(Self {
            plan: plan.fingerprint(),
            binding_layout,
            resource_revisions,
            artifacts,
        })
    }

    pub fn validate(
        &self,
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        artifacts: &ArtifactCatalog,
    ) -> Result<PreparedPlanState, PreparedPlanInvalidation> {
        if self.plan != plan.fingerprint() {
            return Err(PreparedPlanInvalidation::PlanChanged {
                prepared: self.plan,
                current: plan.fingerprint(),
            });
        }
        let current_layout = bindings
            .layout_fingerprint(plan)
            .map_err(PreparedPlanInvalidation::Bindings)?;
        if self.binding_layout != current_layout {
            return Err(PreparedPlanInvalidation::BindingLayoutChanged {
                prepared: self.binding_layout,
                current: current_layout,
            });
        }
        for (kernel, expected) in &self.artifacts {
            let current =
                artifacts
                    .get(kernel)
                    .ok_or_else(|| PreparedPlanInvalidation::ArtifactMissing {
                        kernel: kernel.clone(),
                    })?;
            if *expected != current {
                return Err(PreparedPlanInvalidation::ArtifactChanged {
                    kernel: kernel.clone(),
                    prepared: *expected,
                    current,
                });
            }
        }
        let mut changed = Vec::new();
        for resource in plan.resources() {
            let current = bindings.resources[&resource.id()].revision;
            if self.resource_revisions.get(&resource.id().index()) != Some(&current) {
                changed.push(resource.id());
            }
        }
        if changed.is_empty() {
            Ok(PreparedPlanState::Current)
        } else {
            Ok(PreparedPlanState::NeedsRebind { resources: changed })
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PreparedPlanInvalidation {
    #[error("compiled plan fingerprint changed from {prepared:?} to {current:?}")]
    PlanChanged {
        prepared: PlanFingerprint,
        current: PlanFingerprint,
    },
    #[error("replay bindings are invalid: {0}")]
    Bindings(#[source] ReplayBindingError),
    #[error("binding layout changed from {prepared:?} to {current:?}")]
    BindingLayoutChanged {
        prepared: BindingLayoutFingerprint,
        current: BindingLayoutFingerprint,
    },
    #[error("kernel artifact {kernel:?} is missing")]
    ArtifactMissing { kernel: String },
    #[error("kernel artifact {kernel:?} changed generation or bytes")]
    ArtifactChanged {
        kernel: String,
        prepared: KernelArtifactIdentity,
        current: KernelArtifactIdentity,
    },
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum ReplayBindingError {
    #[error("resource pointer is null")]
    NullResourcePointer,
    #[error("resource binding is empty")]
    EmptyResourceBinding,
    #[error("resource binding of {size} bytes overflows the host address space")]
    ResourceAddressOverflow { size: u64 },
    #[error("resource {resource:?} is not bound")]
    ResourceNotBound { resource: ResourceId },
    #[error("resource {resource:?} requires {required} bytes but binding has {bound}")]
    ResourceTooSmall {
        resource: ResourceId,
        required: u64,
        bound: u64,
    },
    #[error("resources {first:?} and {second:?} alias")]
    AliasingResources {
        first: ResourceId,
        second: ResourceId,
    },
    #[error("dynamic scalar slot {slot:?} is not bound")]
    ScalarNotBound { slot: ScalarSlotId },
    #[error("dynamic scalar slot {slot:?} has {actual} bytes, expected {expected}")]
    ScalarSize {
        slot: ScalarSlotId,
        expected: u32,
        actual: usize,
    },
}

#[cfg(test)]
mod property_tests {
    use std::num::NonZeroUsize;

    use proptest::prelude::*;

    use super::*;
    use crate::{Access, CompileOptions, Dim3, KernelArg, KernelLaunch, Recorder, ReplayMode};

    fn bound_plan(
        resource_size: u64,
        scalar_size: u32,
    ) -> (CompiledPlan, ResourceId, ScalarSlotId) {
        let mut recorder = Recorder::new();
        let resource = recorder
            .resource("activation", resource_size)
            .expect("valid resource");
        let region = recorder
            .region(resource, 0, resource_size)
            .expect("valid region");
        let slot = ScalarSlotId::new(7);
        let launch = KernelLaunch::new(
            "bound-kernel",
            Dim3::x(1).expect("valid grid"),
            Dim3::x(1).expect("valid block"),
        )
        .expect("valid launch")
        .with_arguments([KernelArg::scalar_slot(slot, scalar_size).expect("valid scalar slot")]);
        recorder
            .dispatch(launch, [Access::read(region)])
            .expect("valid dispatch");
        let plan = recorder
            .compile(CompileOptions::new(
                NonZeroUsize::MIN,
                ReplayMode::TokenLatency,
            ))
            .expect("valid plan");
        (plan, resource, slot)
    }

    fn external_binding(storage: &mut [u8]) -> ResourceBinding {
        // Validation only inspects the live allocation's address and extent;
        // these tests never submit the host allocation for GPU replay.
        unsafe {
            ResourceBinding::new(
                storage.as_mut_ptr().cast(),
                storage.len() as u64,
                BindingRevision(1),
                AllocationPolicy::External,
            )
        }
        .expect("non-empty live allocation")
    }

    proptest! {
        #[test]
        fn resource_binding_extent_is_an_exact_boundary(
            required in 1_u64..8_192,
            bound in 1_u64..8_192,
            scalar_size in 1_u32..65,
        ) {
            let (plan, resource, slot) = bound_plan(required, scalar_size);
            let mut storage = vec![0_u8; bound as usize];
            let mut bindings = ReplayBindings::new();
            bindings.bind_resource(resource, external_binding(&mut storage));
            bindings.bind_scalar(slot, vec![0_u8; scalar_size as usize]);

            let result = bindings.validate(&plan);
            if bound >= required {
                prop_assert_eq!(result, Ok(()));
            } else {
                prop_assert_eq!(
                    result,
                    Err(ReplayBindingError::ResourceTooSmall {
                        resource,
                        required,
                        bound,
                    }),
                );
            }
        }

        #[test]
        fn scalar_binding_size_is_an_exact_boundary(
            resource_size in 1_u64..8_192,
            expected in 1_u32..65,
            actual in 0_usize..96,
        ) {
            let (plan, resource, slot) = bound_plan(resource_size, expected);
            let mut storage = vec![0_u8; resource_size as usize];
            let mut bindings = ReplayBindings::new();
            bindings.bind_resource(resource, external_binding(&mut storage));
            bindings.bind_scalar(slot, vec![0_u8; actual]);

            let result = bindings.validate(&plan);
            if actual == expected as usize {
                prop_assert_eq!(result, Ok(()));
            } else {
                prop_assert_eq!(
                    result,
                    Err(ReplayBindingError::ScalarSize {
                        slot,
                        expected,
                        actual,
                    }),
                );
            }
        }
    }
}
