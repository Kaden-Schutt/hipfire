use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use crate::{KernargAbi, KernelArtifactIdentity, RecordError};

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId {
    pub(crate) owner: u64,
    pub(crate) index: u32,
}

impl NodeId {
    /// The stable insertion index within this node's recorder.
    pub fn index(self) -> u32 {
        self.index
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.index)
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceId {
    pub(crate) owner: u64,
    pub(crate) index: u32,
}

impl ResourceId {
    /// The stable insertion index within this resource's recorder.
    pub fn index(self) -> u32 {
        self.index
    }
}

impl fmt::Debug for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResourceId({})", self.index)
    }
}

/// Stable identifier for a scalar value patched at replay time.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScalarSlotId(u32);

impl ScalarSlotId {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A byte range within one logical device allocation.
///
/// Distinct resources are assumed not to alias. A concrete backend must retain
/// that invariant when it binds resource IDs to device allocations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceRegion {
    resource: ResourceId,
    offset: u64,
    len: u64,
}

impl DeviceRegion {
    pub fn new(resource: ResourceId, offset: u64, len: u64) -> Result<Self, RecordError> {
        if len == 0 {
            return Err(RecordError::EmptyRegion);
        }
        offset
            .checked_add(len)
            .ok_or(RecordError::RegionAddressOverflow { offset, len })?;
        Ok(Self {
            resource,
            offset,
            len,
        })
    }

    pub fn resource(self) -> ResourceId {
        self.resource
    }

    pub fn offset(self) -> u64 {
        self.offset
    }

    pub fn len(self) -> u64 {
        self.len
    }

    /// Always false: zero-length regions are rejected at construction.
    pub fn is_empty(self) -> bool {
        false
    }

    pub fn end(self) -> u64 {
        // Construction proves that this does not overflow.
        self.offset + self.len
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.resource == other.resource && self.offset < other.end() && other.offset < self.end()
    }

    pub(crate) fn intersection(self, other: Self) -> Option<Self> {
        if !self.overlaps(other) {
            return None;
        }
        let offset = self.offset.max(other.offset);
        let end = self.end().min(other.end());
        Some(Self {
            resource: self.resource,
            offset,
            len: end - offset,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccessMode {
    Read,
    Write,
}

/// One declared memory effect of a dispatch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Access {
    region: DeviceRegion,
    mode: AccessMode,
}

impl Access {
    pub fn read(region: DeviceRegion) -> Self {
        Self {
            region,
            mode: AccessMode::Read,
        }
    }

    pub fn write(region: DeviceRegion) -> Self {
        Self {
            region,
            mode: AccessMode::Write,
        }
    }

    pub fn region(self) -> DeviceRegion {
        self.region
    }

    pub fn mode(self) -> AccessMode {
        self.mode
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Result<Self, RecordError> {
        if x == 0 || y == 0 || z == 0 {
            return Err(RecordError::ZeroLaunchDimension { x, y, z });
        }
        Ok(Self { x, y, z })
    }

    pub fn x(x: u32) -> Result<Self, RecordError> {
        Self::new(x, 1, 1)
    }
}

/// A backend-neutral kernel argument.
///
/// Scalar bytes are copied into the plan. Resource arguments remain symbolic
/// until a concrete backend binds the plan to device allocations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelArg {
    Scalar(Arc<[u8]>),
    ScalarSlot {
        slot: ScalarSlotId,
        size: u32,
    },
    Resource {
        resource: ResourceId,
        byte_offset: u64,
    },
}

impl KernelArg {
    pub fn scalar(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self::Scalar(bytes.into())
    }

    pub fn resource(resource: ResourceId, byte_offset: u64) -> Self {
        Self::Resource {
            resource,
            byte_offset,
        }
    }

    pub fn scalar_slot(slot: ScalarSlotId, size: u32) -> Result<Self, RecordError> {
        if size == 0 {
            return Err(RecordError::EmptyScalarSlot { slot });
        }
        Ok(Self::ScalarSlot { slot, size })
    }
}

/// Immutable metadata required to resolve and launch one kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelLaunch {
    kernel: Arc<str>,
    grid: Dim3,
    block: Dim3,
    dynamic_shared_bytes: u32,
    arguments: Vec<KernelArg>,
    kernarg_abi: Option<Arc<KernargAbi>>,
    artifact_identity: Option<KernelArtifactIdentity>,
    estimated_work: NonZeroU32,
}

impl KernelLaunch {
    pub fn new(kernel: impl Into<Arc<str>>, grid: Dim3, block: Dim3) -> Result<Self, RecordError> {
        let kernel = kernel.into();
        if kernel.trim().is_empty() {
            return Err(RecordError::EmptyKernelKey);
        }
        Ok(Self {
            kernel,
            grid,
            block,
            dynamic_shared_bytes: 0,
            arguments: Vec::new(),
            kernarg_abi: None,
            artifact_identity: None,
            estimated_work: NonZeroU32::MIN,
        })
    }

    pub fn with_dynamic_shared_bytes(mut self, bytes: u32) -> Self {
        self.dynamic_shared_bytes = bytes;
        self
    }

    pub fn with_arguments(mut self, arguments: impl IntoIterator<Item = KernelArg>) -> Self {
        self.arguments.extend(arguments);
        self
    }

    /// Attach the explicit contiguous kernarg layout used by direct AQL.
    ///
    /// Layout construction validates offsets and overlap. Argument count and
    /// sizes are checked when the recorder compiles, so callers may attach the
    /// layout before or after arguments.
    pub fn with_kernarg_abi(mut self, abi: KernargAbi) -> Self {
        self.kernarg_abi = Some(Arc::new(abi));
        self
    }

    pub fn with_artifact_identity(mut self, identity: KernelArtifactIdentity) -> Self {
        self.artifact_identity = Some(identity);
        self
    }

    /// Set a relative scheduling estimate. This is a deterministic heuristic,
    /// not a claim about measured GPU duration.
    pub fn with_estimated_work(mut self, work: NonZeroU32) -> Self {
        self.estimated_work = work;
        self
    }

    pub fn kernel(&self) -> &str {
        &self.kernel
    }

    pub fn grid(&self) -> Dim3 {
        self.grid
    }

    pub fn block(&self) -> Dim3 {
        self.block
    }

    pub fn dynamic_shared_bytes(&self) -> u32 {
        self.dynamic_shared_bytes
    }

    pub fn arguments(&self) -> &[KernelArg] {
        &self.arguments
    }

    pub fn kernarg_abi(&self) -> Option<&KernargAbi> {
        self.kernarg_abi.as_deref()
    }

    pub fn artifact_identity(&self) -> Option<KernelArtifactIdentity> {
        self.artifact_identity
    }

    pub fn estimated_work(&self) -> NonZeroU32 {
        self.estimated_work
    }
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn region_intersection_matches_half_open_ranges(
            owner in any::<u64>(),
            resource_index in any::<u32>(),
            first_offset in 0_u64..1_000_000,
            first_len in 1_u64..4_096,
            second_offset in 0_u64..1_000_000,
            second_len in 1_u64..4_096,
        ) {
            let resource = ResourceId {
                owner,
                index: resource_index,
            };
            let first = DeviceRegion::new(resource, first_offset, first_len).unwrap();
            let second = DeviceRegion::new(resource, second_offset, second_len).unwrap();
            let expected = first_offset < second_offset + second_len
                && second_offset < first_offset + first_len;

            prop_assert_eq!(first.overlaps(second), expected);
            prop_assert_eq!(second.overlaps(first), expected);

            match first.intersection(second) {
                Some(intersection) => {
                    prop_assert!(expected);
                    prop_assert_eq!(intersection.resource(), resource);
                    prop_assert_eq!(intersection.offset(), first_offset.max(second_offset));
                    prop_assert_eq!(
                        intersection.end(),
                        (first_offset + first_len).min(second_offset + second_len),
                    );
                }
                None => prop_assert!(!expected),
            }
        }

        #[test]
        fn distinct_resources_never_overlap(
            owner in any::<u64>(),
            resource_index in any::<u32>(),
            offset in 0_u64..1_000_000,
            len in 1_u64..4_096,
        ) {
            let first = ResourceId {
                owner,
                index: resource_index,
            };
            let second = ResourceId {
                owner,
                index: resource_index.wrapping_add(1),
            };
            let first = DeviceRegion::new(first, offset, len).unwrap();
            let second = DeviceRegion::new(second, offset, len).unwrap();

            prop_assert!(!first.overlaps(second));
            prop_assert_eq!(first.intersection(second), None);
        }
    }
}
