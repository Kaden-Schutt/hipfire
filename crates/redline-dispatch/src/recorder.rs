use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Access, CompileError, CompileOptions, CompiledPlan, KernelArg, KernelLaunch};
use crate::{DeviceRegion, NodeId, ResourceId};

static NEXT_RECORDER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub(crate) struct Resource {
    pub(crate) label: String,
    pub(crate) size: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) id: NodeId,
    pub(crate) launch: KernelLaunch,
    pub(crate) accesses: Vec<Access>,
}

/// Mutable builder for one dispatch DAG.
pub struct Recorder {
    pub(crate) owner: u64,
    pub(crate) resources: Vec<Resource>,
    pub(crate) nodes: Vec<Node>,
    /// Directed edges are `(prerequisite, dependent)`.
    pub(crate) edges: BTreeSet<(NodeId, NodeId)>,
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Recorder {
    pub fn new() -> Self {
        let owner = NEXT_RECORDER_ID.fetch_add(1, Ordering::Relaxed);
        assert!(owner != u64::MAX, "recorder identity space exhausted");
        Self {
            owner,
            resources: Vec::new(),
            nodes: Vec::new(),
            edges: BTreeSet::new(),
        }
    }

    pub fn resource(
        &mut self,
        label: impl Into<String>,
        size: u64,
    ) -> Result<ResourceId, RecordError> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(RecordError::EmptyResourceLabel);
        }
        if size == 0 {
            return Err(RecordError::EmptyResource);
        }
        let index =
            u32::try_from(self.resources.len()).map_err(|_| RecordError::TooManyResources)?;
        self.resources.push(Resource { label, size });
        Ok(ResourceId {
            owner: self.owner,
            index,
        })
    }

    pub fn region(
        &self,
        resource: ResourceId,
        offset: u64,
        len: u64,
    ) -> Result<DeviceRegion, RecordError> {
        let region = DeviceRegion::new(resource, offset, len)?;
        self.validate_region(region)?;
        Ok(region)
    }

    pub fn dispatch(
        &mut self,
        launch: KernelLaunch,
        accesses: impl IntoIterator<Item = Access>,
    ) -> Result<NodeId, RecordError> {
        let accesses = accesses.into_iter().collect::<Vec<_>>();
        for access in &accesses {
            self.validate_region(access.region())?;
        }

        for argument in launch.arguments() {
            match argument {
                KernelArg::Resource {
                    resource,
                    byte_offset,
                } => {
                    let entry = self.resource_entry(*resource)?;
                    if *byte_offset >= entry.size {
                        return Err(RecordError::ResourceArgumentOutOfBounds {
                            resource: *resource,
                            offset: *byte_offset,
                            size: entry.size,
                        });
                    }
                    if !accesses.iter().any(|access| {
                        let region = access.region();
                        region.resource() == *resource
                            && region.offset() <= *byte_offset
                            && *byte_offset < region.end()
                    }) {
                        return Err(RecordError::UndeclaredResourceArgument {
                            resource: *resource,
                            offset: *byte_offset,
                        });
                    }
                }
                KernelArg::ScalarSlot { slot, size: 0 } => {
                    return Err(RecordError::EmptyScalarSlot { slot: *slot });
                }
                KernelArg::Scalar(_) | KernelArg::ScalarSlot { .. } => {}
            }
        }

        let index = u32::try_from(self.nodes.len()).map_err(|_| RecordError::TooManyNodes)?;
        let id = NodeId {
            owner: self.owner,
            index,
        };
        self.nodes.push(Node {
            id,
            launch,
            accesses,
        });
        Ok(id)
    }

    /// Make `node` wait for `prerequisite`.
    ///
    /// A cycle is rejected without mutating the recorder.
    pub fn depends_on(
        &mut self,
        node: NodeId,
        prerequisite: NodeId,
    ) -> Result<&mut Self, RecordError> {
        self.validate_node(node)?;
        self.validate_node(prerequisite)?;
        if node == prerequisite || self.path_exists(node, prerequisite) {
            return Err(RecordError::DependencyCycle { node, prerequisite });
        }
        self.edges.insert((prerequisite, node));
        Ok(self)
    }

    pub fn compile(&self, options: CompileOptions) -> Result<CompiledPlan, CompileError> {
        CompiledPlan::compile(self, options)
    }

    fn resource_entry(&self, resource: ResourceId) -> Result<&Resource, RecordError> {
        if resource.owner != self.owner {
            return Err(RecordError::ForeignResource(resource));
        }
        self.resources
            .get(resource.index as usize)
            .ok_or(RecordError::UnknownResource(resource))
    }

    fn validate_region(&self, region: DeviceRegion) -> Result<(), RecordError> {
        let resource = self.resource_entry(region.resource())?;
        if region.end() > resource.size {
            return Err(RecordError::RegionOutOfBounds {
                resource: region.resource(),
                offset: region.offset(),
                len: region.len(),
                size: resource.size,
            });
        }
        Ok(())
    }

    fn validate_node(&self, node: NodeId) -> Result<(), RecordError> {
        if node.owner != self.owner {
            return Err(RecordError::ForeignNode(node));
        }
        if self.nodes.get(node.index as usize).is_none() {
            return Err(RecordError::UnknownNode(node));
        }
        Ok(())
    }

    fn path_exists(&self, from: NodeId, to: NodeId) -> bool {
        let mut stack = vec![from];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(node) = stack.pop() {
            if node == to {
                return true;
            }
            if std::mem::replace(&mut seen[node.index as usize], true) {
                continue;
            }
            stack.extend(
                self.edges
                    .iter()
                    .filter_map(|(source, target)| (*source == node).then_some(*target)),
            );
        }
        false
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("kernel key is empty")]
    EmptyKernelKey,
    #[error("launch dimensions must all be nonzero, got ({x}, {y}, {z})")]
    ZeroLaunchDimension { x: u32, y: u32, z: u32 },
    #[error("resource label is empty")]
    EmptyResourceLabel,
    #[error("resource size is zero")]
    EmptyResource,
    #[error("device region size is zero")]
    EmptyRegion,
    #[error("dynamic scalar slot {slot:?} has zero size")]
    EmptyScalarSlot { slot: crate::ScalarSlotId },
    #[error("device region address overflows: offset {offset} + length {len}")]
    RegionAddressOverflow { offset: u64, len: u64 },
    #[error(
        "region {offset}..{end} exceeds {resource:?} size {size}",
        end = offset.saturating_add(*len)
    )]
    RegionOutOfBounds {
        resource: ResourceId,
        offset: u64,
        len: u64,
        size: u64,
    },
    #[error("resource argument offset {offset} exceeds {resource:?} size {size}")]
    ResourceArgumentOutOfBounds {
        resource: ResourceId,
        offset: u64,
        size: u64,
    },
    #[error("resource argument {resource:?}+{offset} is not covered by a declared memory access")]
    UndeclaredResourceArgument { resource: ResourceId, offset: u64 },
    #[error("{0:?} belongs to another recorder")]
    ForeignResource(ResourceId),
    #[error("unknown resource {0:?}")]
    UnknownResource(ResourceId),
    #[error("{0:?} belongs to another recorder")]
    ForeignNode(NodeId),
    #[error("unknown node {0:?}")]
    UnknownNode(NodeId),
    #[error("adding dependency {node:?} <- {prerequisite:?} would create a cycle")]
    DependencyCycle { node: NodeId, prerequisite: NodeId },
    #[error("recorder contains more than u32::MAX resources")]
    TooManyResources,
    #[error("recorder contains more than u32::MAX nodes")]
    TooManyNodes,
}
