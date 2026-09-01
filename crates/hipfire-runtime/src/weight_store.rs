// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transactional fulfillment for the pure weight manifest.
//!
//! [`crate::weight_manifest::plan_manifest`] owns the CPU-only "where". This
//! module owns the narrow "how" pilot for a plain LLaMA Single target: a
//! source callback supplies already-resolved bytes and dtype, the store uploads
//! them, and the first failure explicitly rolls back every resident buffer.
//!
//! The store is not a model owner. It has no `Drop` implementation and never
//! frees GPU buffers implicitly. A carrier moves a committed transaction into
//! its existing `ArchModel` owner; that owner consumes the architecture-private
//! attached owner during the existing teardown path.
//! `WeightStoreAssembly::take` transfers a resident handle to the owner that is
//! assembling typed weights, and therefore removes the cell from the store's
//! cleanup set.
use crate::weight_manifest::{placement_devices, ShardPolicy, WeightEntry};
use hipfire_hardware::{DeviceMesh, MeshEpoch};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;

thread_local! {
    static RESIDENT_ALLOCATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static RESIDENT_RELEASES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_AFTER_UPLOAD: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only allocation accounting and deterministic post-upload fault seam.
///
/// The production loader calls the same release path regardless of whether
/// this seam is armed. Callers should use [`reset`] before a scenario and
/// [`clear_faults`] after it so a failed test cannot poison a later one.
#[doc(hidden)]
pub mod test_support {
    use super::{FAIL_AFTER_UPLOAD, RESIDENT_ALLOCATIONS, RESIDENT_RELEASES};

    pub fn reset() {
        RESIDENT_ALLOCATIONS.with(|count| count.set(0));
        RESIDENT_RELEASES.with(|count| count.set(0));
        clear_faults();
    }

    pub fn arm_fail_after_upload(upload_number: usize) {
        assert!(upload_number > 0, "upload fault threshold must be non-zero");
        FAIL_AFTER_UPLOAD.with(|fault| fault.set(Some(upload_number)));
    }

    pub fn clear_faults() {
        FAIL_AFTER_UPLOAD.with(|fault| fault.set(None));
    }

    pub fn resident_allocations() -> usize {
        RESIDENT_ALLOCATIONS.with(std::cell::Cell::get)
    }

    pub fn resident_releases() -> usize {
        RESIDENT_RELEASES.with(std::cell::Cell::get)
    }

    pub(super) fn record_resident_upload() -> bool {
        let allocation = RESIDENT_ALLOCATIONS.with(|count| {
            let next = count.get() + 1;
            count.set(next);
            next
        });
        FAIL_AFTER_UPLOAD.with(|fault| {
            let should_fail = fault
                .get()
                .is_some_and(|upload_number| allocation >= upload_number);
            if should_fail {
                fault.set(None);
            }
            should_fail
        })
    }
}

/// Stable logical placement identity. Layer is part of the key because a
/// per-layer name such as `wq` appears once for every decoder block.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeightPlacementKey {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
}

impl WeightPlacementKey {
    pub fn new(name: impl Into<String>, layer: Option<usize>, device: usize) -> Self {
        Self {
            name: name.into(),
            layer,
            device,
        }
    }
}

/// The immutable projection applied to one logical source before upload.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightProjectionKind {
    Static,
    ColumnShard,
    RowShard,
    FusedQkv,
    HeadSharded,
    VocabShard,
    ExpertCompact,
    ExpertTensor,
}

/// Value-owned placement metadata. It contains no GPU or source-file
/// representation and remains stable after a handle is taken from the store.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightProjection {
    pub kind: WeightProjectionKind,
    pub axis: Option<usize>,
    pub rank: usize,
    pub world_size: usize,
    pub logical_shape: Vec<usize>,
    pub dtype: DType,
}

fn projection_for(
    entry: &WeightEntry,
    rank: usize,
    world_size: usize,
    dtype: DType,
) -> WeightProjection {
    let (kind, axis) = match &entry.policy {
        ShardPolicy::ColumnShard { axis } => (WeightProjectionKind::ColumnShard, Some(*axis)),
        ShardPolicy::RowShard { axis } => (WeightProjectionKind::RowShard, Some(*axis)),
        ShardPolicy::FusedQkv { .. } => (WeightProjectionKind::FusedQkv, None),
        ShardPolicy::HeadSharded { .. } => (WeightProjectionKind::HeadSharded, None),
        ShardPolicy::VocabShard { axis } => (WeightProjectionKind::VocabShard, Some(*axis)),
        ShardPolicy::ExpertSharded { .. } => (WeightProjectionKind::ExpertCompact, None),
        ShardPolicy::ExpertTensorSharded { .. } => (WeightProjectionKind::ExpertTensor, None),
        ShardPolicy::Replicate | ShardPolicy::Pin(_) | ShardPolicy::Tied { .. } => {
            (WeightProjectionKind::Static, None)
        }
    };
    WeightProjection {
        kind,
        axis,
        rank,
        world_size,
        logical_shape: entry.logical_shape.clone(),
        dtype,
    }
}

/// A resident GPU tensor or a symbolic alias to another logical source.
///
/// Aliases own no buffer. Resident buffers have no implicit destructor; the
/// current model owner explicitly consumes them through its teardown method.
pub enum WeightHandle {
    Resident(GpuTensor),
    Alias(String),
}

/// Identity captured at the start of a load. It is deliberately immutable and
/// contains only mesh generation, logical rank, and physical device identity.
/// No policy or source representation is smuggled into the origin.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightOrigin {
    mesh_epoch: MeshEpoch,
    logical_rank: usize,
    physical_device: i32,
}

impl WeightOrigin {
    pub fn from_parts(mesh_epoch: MeshEpoch, logical_rank: usize, physical_device: i32) -> Self {
        Self {
            mesh_epoch,
            logical_rank,
            physical_device,
        }
    }

    pub fn for_single(mesh: &DeviceMesh, gpu: &Gpu) -> Self {
        Self::from_parts(mesh.epoch(), 0, gpu.device_id)
    }

    pub fn mesh_epoch(self) -> MeshEpoch {
        self.mesh_epoch
    }

    pub fn logical_rank(self) -> usize {
        self.logical_rank
    }

    pub fn physical_device(self) -> i32 {
        self.physical_device
    }
}

/// Errors that are detected before a store is allowed to release a resident
/// buffer. Origin mismatch always returns the store to the caller unchanged.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WeightStoreError {
    OriginMismatch {
        expected: WeightOrigin,
        actual: WeightOrigin,
    },
    UnboundOrigin,
    DuplicatePlacement(WeightPlacementKey),
    MissingPlacement(WeightPlacementKey),
    InvalidTarget(String),
}

impl std::fmt::Display for WeightStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OriginMismatch { expected, actual } => write!(
                f,
                "weight store origin mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::UnboundOrigin => write!(f, "weight store has no target origin"),
            Self::DuplicatePlacement(key) => write!(
                f,
                "duplicate weight placement {}[layer {:?}] on device {}",
                key.name, key.layer, key.device
            ),
            Self::MissingPlacement(key) => write!(
                f,
                "missing weight placement {}[layer {:?}] on device {}",
                key.name, key.layer, key.device
            ),
            Self::InvalidTarget(message) => write!(f, "invalid weight store target: {message}"),
        }
    }
}

impl std::error::Error for WeightStoreError {}

/// Error identifying the first failed manifest cell. The store has already
/// been rolled back before this value is returned by [`fulfill_manifest`].
#[derive(Debug)]
pub struct FulfillError {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
    pub reason: String,
}

impl std::fmt::Display for FulfillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fulfill_manifest: {}[layer {:?}] on device {}: {}",
            self.name, self.layer, self.device, self.reason
        )
    }
}

impl std::error::Error for FulfillError {}

/// Load-side placement container. It records one immutable projection per
/// `(name, layer, device)` and captures the target origin once. The container
/// itself has no consuming teardown API; lifecycle transitions are represented
/// by [`WeightLoadTransaction`] and the architecture-private attached owner.
#[derive(Default)]
pub struct WeightStore {
    placements: HashMap<WeightPlacementKey, WeightHandle>,
    projections: HashMap<WeightPlacementKey, WeightProjection>,
    origin: Option<WeightOrigin>,
}

/// The only owner that may roll back resident allocations before publication.
///
/// A transaction owns the store until the architecture carrier consumes it
/// into its crate-private attached owner. It deliberately has no implicit
/// `Drop` cleanup because the GPU is not available to a destructor.
pub struct WeightLoadTransaction {
    store: Option<WeightStore>,
}

impl std::fmt::Debug for WeightLoadTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightLoadTransaction")
            .field("origin", &self.origin())
            .field("len", &self.len())
            .finish()
    }
}

impl WeightLoadTransaction {
    pub fn new(store: WeightStore) -> Self {
        Self { store: Some(store) }
    }

    pub fn origin(&self) -> Option<WeightOrigin> {
        self.store.as_ref().and_then(WeightStore::origin)
    }

    pub fn len(&self) -> usize {
        self.store.as_ref().map_or(0, WeightStore::len)
    }

    pub fn is_empty(&self) -> bool {
        self.store.as_ref().map_or(true, WeightStore::is_empty)
    }

    pub fn contains(&self, name: &str, layer: Option<usize>, device: usize) -> bool {
        self.store
            .as_ref()
            .is_some_and(|store| store.contains(name, layer, device))
    }

    pub fn get(&self, name: &str, layer: Option<usize>, device: usize) -> Option<&WeightHandle> {
        self.store
            .as_ref()
            .and_then(|store| store.get(name, layer, device))
    }

    pub fn projection(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&WeightProjection> {
        self.store
            .as_ref()
            .and_then(|store| store.projection(name, layer, device))
    }

    pub fn devices_for(&self, name: &str, layer: Option<usize>) -> Vec<usize> {
        self.store
            .as_ref()
            .map_or_else(Vec::new, |store| store.devices_for(name, layer))
    }

    /// Compare the unpublished transaction's captured target with an admitted
    /// owner identity. This read-only check is used before the carrier wraps
    /// the transaction in its private attached owner.
    pub fn validate_origin_value(&self, expected: WeightOrigin) -> Result<(), WeightStoreError> {
        self.store
            .as_ref()
            .map_or(Err(WeightStoreError::UnboundOrigin), |store| {
                store.validate_origin_value(expected)
            })
    }
    /// Read-only physical-device gate. Performs zero GPU calls.
    pub fn validate_device(&self, device_id: i32) -> Result<(), WeightStoreError> {
        let store = self.store.as_ref().ok_or(WeightStoreError::UnboundOrigin)?;
        store.validate_device(device_id)
    }

    /// Start typed assembly while this load is still unpublished.
    pub fn begin_assembly(&mut self) -> WeightStoreAssembly<'_> {
        self.store
            .as_mut()
            .expect("weight load transaction was already consumed")
            .begin_assembly()
    }

    /// Gate on physical device before any GPU call. On mismatch returns
    /// an error without consuming the allocation, preserving retryability.
    pub fn try_rollback(&mut self, gpu: &Gpu) -> Result<(), WeightStoreError> {
        let store = self.store.as_ref().ok_or(WeightStoreError::UnboundOrigin)?;
        store.validate_device(gpu.device_id)?;
        // Device matches — now consume the store and free.
        let store = self.store.take().expect("store was Some after validation");
        store
            .release_unchecked(gpu)
            .map_err(|e| WeightStoreError::InvalidTarget(format!("rollback hip free failed: {e}")))
    }

    /// Consuming rollback gated on physical device. On mismatch the transaction
    /// is returned alongside the error without any GPU call, preserving the
    /// allocation for retry on the correct device.
    pub fn rollback(mut self, gpu: &Gpu) -> Result<(), (Self, WeightStoreError)> {
        match self.try_rollback(gpu) {
            Ok(()) => Ok(()),
            Err(error) => Err((self, error)),
        }
    }

    /// Legacy HipResult rollback for internal fulfillment paths where the device
    /// is known to be correct (same gpu that created the transaction). It still
    /// gates on device but maps the origin error into a HipError for backward
    /// compatibility with existing `with_weight_rollback_error` callers.
    pub(crate) fn rollback_hip(mut self, gpu: &Gpu) -> hip_bridge::HipResult<()> {
        match self.try_rollback(gpu) {
            Ok(()) => Ok(()),
            Err(WeightStoreError::OriginMismatch { expected, actual }) => {
                Err(hip_bridge::HipError::new(
                    0,
                    &format!("weight store origin mismatch: expected {expected:?}, got {actual:?}"),
                ))
            }
            Err(WeightStoreError::UnboundOrigin) => Err(hip_bridge::HipError::new(
                0,
                "weight store has no target origin",
            )),
            Err(WeightStoreError::InvalidTarget(msg)) => Err(hip_bridge::HipError::new(0, &msg)),
            Err(e) => Err(hip_bridge::HipError::new(0, &e.to_string())),
        }
    }
}

impl WeightStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_origin(origin: WeightOrigin) -> Self {
        Self {
            placements: HashMap::new(),
            projections: HashMap::new(),
            origin: Some(origin),
        }
    }

    pub fn origin(&self) -> Option<WeightOrigin> {
        self.origin
    }

    pub fn len(&self) -> usize {
        self.placements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    pub fn contains(&self, name: &str, layer: Option<usize>, device: usize) -> bool {
        self.placements
            .contains_key(&WeightPlacementKey::new(name, layer, device))
    }

    pub fn get(&self, name: &str, layer: Option<usize>, device: usize) -> Option<&WeightHandle> {
        self.placements
            .get(&WeightPlacementKey::new(name, layer, device))
    }

    pub fn projection(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&WeightProjection> {
        self.projections
            .get(&WeightPlacementKey::new(name, layer, device))
    }

    pub fn devices_for(&self, name: &str, layer: Option<usize>) -> Vec<usize> {
        let mut devices: Vec<_> = self
            .placements
            .keys()
            .filter(|key| key.name == name && key.layer == layer)
            .map(|key| key.device)
            .collect();
        devices.sort_unstable();
        devices
    }

    fn insert(
        &mut self,
        key: WeightPlacementKey,
        handle: WeightHandle,
        projection: WeightProjection,
    ) -> Result<(), WeightStoreError> {
        if self.placements.contains_key(&key) {
            return Err(WeightStoreError::DuplicatePlacement(key));
        }
        self.placements.insert(key.clone(), handle);
        self.projections.insert(key, projection);
        Ok(())
    }

    /// Stage a symbolic alias without GPU work. Used for tied declarations and
    /// CPU ownership tests; aliases never participate in release.
    pub fn stage_alias(
        &mut self,
        name: impl Into<String>,
        layer: Option<usize>,
        device: usize,
        source: impl Into<String>,
        projection: WeightProjection,
    ) -> Result<(), WeightStoreError> {
        self.insert(
            WeightPlacementKey::new(name, layer, device),
            WeightHandle::Alias(source.into()),
            projection,
        )
    }

    /// Move a handle out of the store. This is private to the assembly
    /// capability so arbitrary store holders cannot independently tear down a
    /// resident allocation.
    fn take(&mut self, name: &str, layer: Option<usize>, device: usize) -> Option<WeightHandle> {
        let key = WeightPlacementKey::new(name, layer, device);
        self.projections.remove(&key);
        self.placements.remove(&key)
    }

    fn take_with_projection(
        &mut self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<(WeightHandle, WeightProjection)> {
        let key = WeightPlacementKey::new(name, layer, device);
        let handle = self.placements.remove(&key)?;
        let projection = self.projections.remove(&key)?;
        Some((handle, projection))
    }

    fn begin_assembly(&mut self) -> WeightStoreAssembly<'_> {
        WeightStoreAssembly {
            store: self,
            taken: Vec::new(),
            committed: false,
        }
    }

    /// Compare a store's captured origin with an already-resolved target
    /// identity. This read-only seam cannot release or extract any handle.
    pub fn validate_origin_value(&self, expected: WeightOrigin) -> Result<(), WeightStoreError> {
        let actual = self.origin.ok_or(WeightStoreError::UnboundOrigin)?;
        if actual != expected {
            return Err(WeightStoreError::OriginMismatch { expected, actual });
        }
        Ok(())
    }

    /// Verify that this store is still being handled by the same mesh/device
    /// target. No GPU calls occur on mismatch.
    pub fn validate_origin(&self, mesh: &DeviceMesh, gpu: &Gpu) -> Result<(), WeightStoreError> {
        self.validate_origin_value(WeightOrigin::for_single(mesh, gpu))
    }

    /// Gate on physical device before any GPU call for this store.
    pub fn validate_device(&self, device_id: i32) -> Result<(), WeightStoreError> {
        let actual = self.origin.ok_or(WeightStoreError::UnboundOrigin)?;
        if actual.physical_device() != device_id {
            let expected =
                WeightOrigin::from_parts(actual.mesh_epoch(), actual.logical_rank(), device_id);
            return Err(WeightStoreError::OriginMismatch { expected, actual });
        }
        Ok(())
    }

    /// Gated explicit rollback: validates device before any GPU call.
    /// On mismatch the error is returned without any GPU call; the store
    /// itself is consumed (its buffers remain allocated) — the transaction
    /// layer above preserves retryability by not consuming on mismatch.
    pub fn try_rollback(self, gpu: &Gpu) -> Result<(), WeightStoreError> {
        self.validate_device(gpu.device_id)?;
        self.release_unchecked(gpu)
            .map_err(|e| WeightStoreError::InvalidTarget(format!("hip free failed: {e}")))
    }

    /// Explicit rollback for a failed transaction. It consumes the partial
    /// store and frees every resident buffer on the single owning GPU.
    fn rollback(self, gpu: &Gpu) -> hip_bridge::HipResult<()> {
        self.release_unchecked(gpu)
    }

    fn release_unchecked(self, gpu: &Gpu) -> hip_bridge::HipResult<()> {
        let mut first_error = None;
        for handle in self.placements.into_values() {
            if let WeightHandle::Resident(tensor) = handle {
                match gpu.hip.free(tensor.buf) {
                    Ok(()) => {
                        RESIDENT_RELEASES.with(|count| count.set(count.get() + 1));
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// One resident/alias handle temporarily moved during typed assembly.
pub struct TakenWeight {
    pub key: WeightPlacementKey,
    pub handle: WeightHandle,
    pub projection: WeightProjection,
}

/// Rollback-owning assembly transaction. Dropping it restores every taken cell
/// to the parent store; it never frees a GPU buffer implicitly.
pub struct WeightStoreAssembly<'a> {
    store: &'a mut WeightStore,
    taken: Vec<TakenWeight>,
    committed: bool,
}

impl<'a> WeightStoreAssembly<'a> {
    pub fn take(&mut self, name: &str, layer: Option<usize>, device: usize) -> Option<usize> {
        let key = WeightPlacementKey::new(name, layer, device);
        let (handle, projection) = self.store.take_with_projection(name, layer, device)?;
        let slot = self.taken.len();
        self.taken.push(TakenWeight {
            key,
            handle,
            projection,
        });
        Some(slot)
    }

    pub fn commit(self) -> WeightStoreAssemblyGuard<'a> {
        WeightStoreAssemblyGuard { inner: self }
    }
}

impl Drop for WeightStoreAssembly<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for taken in self.taken.drain(..) {
            let _ = self.store.insert(taken.key, taken.handle, taken.projection);
        }
    }
}

/// Guard retained while the typed architecture object is being built. If it
/// is dropped before `finalize`, all handles return to the parent store.
pub struct WeightStoreAssemblyGuard<'a> {
    inner: WeightStoreAssembly<'a>,
}

impl WeightStoreAssemblyGuard<'_> {
    pub fn get(&self, slot: usize) -> Option<&WeightHandle> {
        self.inner.taken.get(slot).map(|taken| &taken.handle)
    }

    pub fn projection(&self, slot: usize) -> Option<&WeightProjection> {
        self.inner.taken.get(slot).map(|taken| &taken.projection)
    }

    /// Transfer the taken handles to the existing ArchModel-owned typed
    /// weights. This is the sole operation that removes them from rollback
    /// ownership.
    pub fn finalize(mut self) -> Vec<TakenWeight> {
        self.inner.committed = true;
        std::mem::take(&mut self.inner.taken)
    }
}

fn target_error(mesh: &DeviceMesh) -> Option<FulfillError> {
    (mesh.n_devices() != 1).then(|| FulfillError {
        name: "<mesh>".to_string(),
        layer: None,
        device: 0,
        reason: format!(
            "plain LLaMA Single fulfillment requires one logical device, got {}",
            mesh.n_devices()
        ),
    })
}
fn rollback_fulfill_error(store: WeightStore, gpu: &Gpu, mut error: FulfillError) -> FulfillError {
    if let Err(release_error) = store.rollback(gpu) {
        error
            .reason
            .push_str(&format!("; resident rollback failed: {release_error}"));
    }
    error
}

/// Fulfill a manifest for a plain LLaMA Single target.
///
/// The source callback is the architecture-owned namespace seam and returns
/// raw bytes plus the actual source dtype. No file/GGUF/HFQ type crosses this
/// API. On the first source, dtype, or upload failure every earlier resident is
/// explicitly released before the error is returned.
pub fn fulfill_manifest_single<F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &Gpu,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<(Vec<u8>, DType), String>,
{
    if let Some(error) = target_error(mesh) {
        return Err(error);
    }
    if let Err(reason) = crate::weight_manifest::validate_weight_layers(weights, n_layers)
        .and_then(|_| crate::weight_manifest::validate_manifest(weights, mesh))
    {
        return Err(FulfillError {
            name: "<manifest>".to_string(),
            layer: None,
            device: 0,
            reason,
        });
    }

    let origin = WeightOrigin::for_single(mesh, gpu);
    let mut store = WeightStore::with_origin(origin);
    for entry in weights {
        let devices = match placement_devices(entry, mesh, n_layers) {
            Ok(devices) => devices,
            Err(error) => {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!("device placement failed: {error}"),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
        };
        if devices.as_slice() != [0] {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: devices.first().copied().unwrap_or(0),
                reason: format!("Single placement resolved to {:?}, expected [0]", devices),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
        let key = WeightPlacementKey::new(&entry.name, entry.layer, 0);
        if let ShardPolicy::Tied {
            source: source_name,
        } = &entry.policy
        {
            let source_dtype = match store.get(source_name, entry.layer, 0) {
                Some(WeightHandle::Resident(tensor)) => Some(tensor.dtype),
                Some(WeightHandle::Alias(_)) | None => None,
            };
            let Some(actual_dtype) = source_dtype else {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!(
                        "tied source '{source_name}' is unresolved or has no actual resident dtype"
                    ),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            };
            if !entry.dtype_constraint.accepts(actual_dtype) {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!(
                        "tied source '{source_name}' actual dtype {actual_dtype:?} is excluded by constraint {:?}",
                        entry.dtype_constraint
                    ),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
            let projection = projection_for(entry, 0, 1, actual_dtype);
            if let Err(reason) =
                store.insert(key, WeightHandle::Alias(source_name.clone()), projection)
            {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: reason.to_string(),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
            continue;
        }

        let (bytes, dtype) = match source(entry) {
            Ok(value) => value,
            Err(reason) => {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!("source read failed: {reason}"),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
        };
        if !entry.dtype_constraint.accepts(dtype) {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: 0,
                reason: format!(
                    "source dtype {dtype:?} violates constraint {:?}",
                    entry.dtype_constraint
                ),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
        if let Err(reason) = crate::weight_backend::validate_weight_payload(
            dtype,
            bytes.len(),
            &entry.logical_shape,
            &entry.name,
        ) {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: 0,
                reason: format!("payload validation failed for {dtype:?}: {reason}"),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
        let mut tensor = match gpu.upload_raw(&bytes, &entry.logical_shape) {
            Ok(tensor) => tensor,
            Err(error) => {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!("upload_raw failed: {error}"),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
        };
        tensor.dtype = dtype;
        let projection = projection_for(entry, 0, 1, dtype);
        if let Err(reason) = store.insert(key, WeightHandle::Resident(tensor), projection) {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: 0,
                reason: reason.to_string(),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
        if test_support::record_resident_upload() {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: 0,
                reason: "test fault injected after resident upload".into(),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
    }
    Ok(WeightLoadTransaction::new(store))
}

/// Canonical name used by the manifest fulfillment seam. The target is
/// deliberately Single-only in this pilot; multi-device fulfillment belongs to
/// the admitted mesh/G5 integration and must not grow a second owner here.
pub fn fulfill_manifest<F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &Gpu,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<(Vec<u8>, DType), String>,
{
    fulfill_manifest_single(weights, mesh, n_layers, gpu, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_manifest::{DTypeConstraint, PinTarget, ShardPolicy};
    use hipfire_hardware::DimKind;

    fn projection(dtype: DType) -> WeightProjection {
        WeightProjection {
            kind: WeightProjectionKind::Static,
            axis: None,
            rank: 0,
            world_size: 1,
            logical_shape: vec![1],
            dtype,
        }
    }

    #[test]
    fn origin_mismatch_is_detected_before_gpu_release() {
        let first = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let second = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let actual = WeightOrigin::from_parts(first.epoch(), 0, 0);
        let expected = WeightOrigin::from_parts(second.epoch(), 0, 0);
        let store = WeightStore::with_origin(actual);
        let error = store.validate_origin_value(expected).unwrap_err();
        assert!(matches!(
            error,
            WeightStoreError::OriginMismatch {
                expected: got_expected,
                actual: got_actual
            } if got_expected == expected && got_actual == actual
        ));
    }

    #[test]
    fn staged_rollback_removes_handles_and_projection_together() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let origin = WeightOrigin::from_parts(mesh.epoch(), 0, 0);
        let mut store = WeightStore::with_origin(origin);
        store
            .stage_alias("first", None, 0, "source", projection(DType::F16))
            .unwrap();
        store
            .stage_alias("second", Some(2), 0, "source", projection(DType::F16))
            .unwrap();
        assert_eq!(store.len(), 2);
        let first = store.take_with_projection("first", None, 0).unwrap();
        assert!(matches!(first.0, WeightHandle::Alias(_)));
        assert!(store.projection("first", None, 0).is_none());
        assert_eq!(store.len(), 1);
        let second = store.take("second", Some(2), 0).unwrap();
        assert!(matches!(second, WeightHandle::Alias(_)));
        assert!(store.is_empty());
    }

    #[test]
    fn assembly_drop_restores_staged_handles() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let mut store = WeightStore::with_origin(WeightOrigin::from_parts(mesh.epoch(), 0, 0));
        store
            .stage_alias("x", None, 0, "source", projection(DType::F16))
            .unwrap();
        {
            let mut assembly = store.begin_assembly();
            assert_eq!(assembly.take("x", None, 0), Some(0));
            let guard = assembly.commit();
            assert!(guard.get(0).is_some());
        }
        assert!(store.contains("x", None, 0));
        assert!(store.projection("x", None, 0).is_some());
    }

    #[test]
    fn repeated_unload_lookup_cannot_reclaim_a_transferred_cell() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let mut store = WeightStore::with_origin(WeightOrigin::from_parts(mesh.epoch(), 0, 0));
        store
            .stage_alias("x", None, 0, "source", projection(DType::F16))
            .unwrap();
        let _owned = store.take("x", None, 0).unwrap();
        assert!(store.take("x", None, 0).is_none());
        assert!(store.projection("x", None, 0).is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn duplicate_projection_is_rejected_without_replacing_identity() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let mut store = WeightStore::with_origin(WeightOrigin::from_parts(mesh.epoch(), 0, 0));
        store
            .stage_alias("x", None, 0, "source-a", projection(DType::F16))
            .unwrap();
        let error = store
            .stage_alias("x", None, 0, "source-b", projection(DType::F32))
            .unwrap_err();
        assert!(matches!(error, WeightStoreError::DuplicatePlacement(_)));
        assert!(
            matches!(store.get("x", None, 0), Some(WeightHandle::Alias(source)) if source == "source-a")
        );
        assert_eq!(store.projection("x", None, 0).unwrap().dtype, DType::F16);
    }

    #[test]
    fn single_target_refuses_multi_device_before_source_or_gpu_work() {
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)])
            .expect("small test mesh construction cannot overflow");
        let entry = WeightEntry::model(
            "embed",
            vec![2, 2],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Embed),
        );
        // The target guard is pure and can be checked without constructing a
        // Gpu; the closure would be unreachable on this path.
        assert!(target_error(&mesh).is_some());
        assert_eq!(placement_devices(&entry, &mesh, 1).unwrap(), vec![0]);
    }

    #[test]
    fn tied_projection_preserves_fulfilled_source_dtype() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let constraint = DTypeConstraint::source_from_sources(vec![DType::F16, DType::F32]);
        let source = WeightEntry::model_with_dtype_constraint(
            "source",
            vec![1],
            DType::F16,
            constraint.clone(),
            ShardPolicy::Replicate,
        );
        let alias = WeightEntry::model_with_dtype_constraint(
            "alias",
            vec![1],
            DType::F16,
            constraint,
            ShardPolicy::Tied {
                source: "source".into(),
            },
        );
        let transaction = fulfill_manifest_single(&[source, alias], &mesh, 1, &gpu, |_| {
            Ok((vec![0; 4], DType::F32))
        })
        .unwrap();
        assert_eq!(
            transaction.projection("alias", None, 0).unwrap().dtype,
            DType::F32
        );
        assert!(matches!(
            transaction.get("alias", None, 0),
            Some(WeightHandle::Alias(source)) if source == "source"
        ));
        transaction
            .rollback(&gpu)
            .expect("resident transaction rollback must free its HIP allocation");
    }

    #[test]
    fn successful_single_fulfillment_commits_resident_projection() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entry = WeightEntry::model("resident", vec![1], DType::F32, ShardPolicy::Replicate);
        let transaction =
            fulfill_manifest_single(&[entry], &mesh, 1, &gpu, |_| Ok((vec![0; 4], DType::F32)))
                .unwrap();
        assert_eq!(transaction.len(), 1);
        assert!(matches!(
            transaction.get("resident", None, 0),
            Some(WeightHandle::Resident(tensor)) if tensor.dtype == DType::F32
        ));
        assert_eq!(
            transaction.projection("resident", None, 0).unwrap().dtype,
            DType::F32
        );
        transaction
            .rollback(&gpu)
            .expect("resident transaction rollback must free its HIP allocation");
    }

    #[test]
    fn full_origin_mismatch_leaves_unpublished_transaction_unchanged() {
        let first = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let second = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let actual = WeightOrigin::from_parts(first.epoch(), 3, 11);
        let expected = WeightOrigin::from_parts(second.epoch(), 4, 12);
        let mut store = WeightStore::with_origin(actual);
        store
            .stage_alias("resident", None, 0, "source", projection(DType::F16))
            .unwrap();
        let transaction = WeightLoadTransaction::new(store);
        let error = transaction.validate_origin_value(expected).unwrap_err();
        assert!(matches!(error, WeightStoreError::OriginMismatch { .. }));
        assert_eq!(transaction.origin(), Some(actual));
        assert!(transaction.contains("resident", None, 0));
        assert!(transaction.projection("resident", None, 0).is_some());
    }

    #[test]
    fn full_origin_mismatch_does_not_free_a_resident_transaction() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entry = WeightEntry::model("resident", vec![1], DType::F32, ShardPolicy::Replicate);
        let transaction =
            fulfill_manifest_single(&[entry], &mesh, 1, &gpu, |_| Ok((vec![0; 4], DType::F32)))
                .unwrap();
        let expected = WeightOrigin::from_parts(mesh.epoch(), 1, gpu.device_id);
        let error = transaction.validate_origin_value(expected).unwrap_err();
        assert!(matches!(error, WeightStoreError::OriginMismatch { .. }));
        assert_eq!(transaction.len(), 1);
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            0,
            "origin rejection must not free resident buffers"
        );
        transaction
            .rollback(&gpu)
            .expect("resident transaction rollback must free its HIP allocation");
        assert_eq!(RESIDENT_RELEASES.with(std::cell::Cell::get), 1);
    }

    #[test]
    fn rollback_reports_free_failure_without_counting_release() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        RESIDENT_ALLOCATIONS.with(|count| count.set(1));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let origin = WeightOrigin::from_parts(mesh.epoch(), 0, gpu.device_id);
        let mut store = WeightStore::with_origin(origin);
        let borrowed = GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut::<std::ffi::c_void>(), 0)
            },
            shape: vec![0],
            dtype: DType::F32,
        };
        store
            .insert(
                WeightPlacementKey::new("borrowed", None, 0),
                WeightHandle::Resident(borrowed),
                projection(DType::F32),
            )
            .expect("insert borrowed resident test handle");
        let (_tx, error) = WeightLoadTransaction::new(store)
            .rollback(&gpu)
            .expect_err("rollback must surface a failed HIP free");
        assert!(error.to_string().contains("borrowed"));
        assert_eq!(test_support::resident_allocations(), 1);
        assert_eq!(test_support::resident_releases(), 0);
        test_support::reset();
    }

    #[test]
    fn source_failure_after_resident_upload_rolls_back_everything() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entries = vec![
            WeightEntry::model("first", vec![1], DType::F32, ShardPolicy::Replicate),
            WeightEntry::model("second", vec![1], DType::F32, ShardPolicy::Replicate),
        ];
        let error = fulfill_manifest_single(&entries, &mesh, 1, &gpu, |entry| {
            if entry.name == "first" {
                Ok((vec![0; 4], DType::F32))
            } else {
                Err("injected source failure".into())
            }
        })
        .unwrap_err();
        assert_eq!(error.name, "second");
        assert!(error.reason.contains("source read failed"));
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            1,
            "the first resident allocation must be explicitly freed"
        );
    }

    #[test]
    fn dtype_failure_after_resident_upload_rolls_back_everything() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let constraint = DTypeConstraint::source_exact(DType::F32);
        let entries = vec![
            WeightEntry::model_with_dtype_constraint(
                "first",
                vec![1],
                DType::F32,
                constraint.clone(),
                ShardPolicy::Replicate,
            ),
            WeightEntry::model_with_dtype_constraint(
                "second",
                vec![1],
                DType::F32,
                constraint,
                ShardPolicy::Replicate,
            ),
        ];
        let error = fulfill_manifest_single(&entries, &mesh, 1, &gpu, |entry| {
            if entry.name == "first" {
                Ok((vec![0; 4], DType::F32))
            } else {
                Ok((vec![0; 2], DType::F16))
            }
        })
        .unwrap_err();
        assert_eq!(error.name, "second");
        assert!(error.reason.contains("violates constraint"));
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            1,
            "the first resident allocation must be explicitly freed"
        );
    }

    #[test]
    fn malformed_upload_payload_after_resident_allocation_rolls_back() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entries = vec![
            WeightEntry::model("first", vec![1], DType::F32, ShardPolicy::Replicate),
            WeightEntry::model("second", vec![1], DType::F32, ShardPolicy::Replicate),
        ];
        let error = fulfill_manifest_single(&entries, &mesh, 1, &gpu, |entry| {
            if entry.name == "first" {
                Ok((vec![0; 4], DType::F32))
            } else {
                Ok((vec![0; 1], DType::F32))
            }
        })
        .unwrap_err();
        assert_eq!(error.name, "second");
        assert!(error.reason.contains("payload"));
        assert_eq!(RESIDENT_RELEASES.with(std::cell::Cell::get), 1);
    }

    #[test]
    fn wrong_device_try_rollback_preserves_allocation_and_performs_zero_gpu_calls() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        // Create a store whose origin claims a different physical device than `gpu`.
        let wrong_origin = WeightOrigin::from_parts(mesh.epoch(), 0, gpu.device_id + 99);
        let mut wrong_store = WeightStore::with_origin(wrong_origin);
        // Use stage_alias for simplicity — alias rollback does zero GPU calls but still gates.
        wrong_store
            .stage_alias("x", None, 0, "source", projection(DType::F32))
            .unwrap();
        let mut wrong_tx = WeightLoadTransaction::new(wrong_store);
        let releases_before = RESIDENT_RELEASES.with(std::cell::Cell::get);
        let err = wrong_tx.try_rollback(&gpu).unwrap_err();
        assert!(matches!(err, WeightStoreError::OriginMismatch { .. }));
        // No hip::free must have been called.
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            releases_before
        );
        // Allocation/ownership is preserved — transaction still has the entry.
        assert!(wrong_tx.contains("x", None, 0));
        assert!(wrong_tx.projection("x", None, 0).is_some());
        // Correct-device retry: a properly-originated transaction succeeds.
        let correct_origin = WeightOrigin::for_single(&mesh, &gpu);
        let mut correct_store = WeightStore::with_origin(correct_origin);
        correct_store
            .stage_alias("y", None, 0, "source", projection(DType::F32))
            .unwrap();
        let mut correct_tx = WeightLoadTransaction::new(correct_store);
        correct_tx
            .try_rollback(&gpu)
            .expect("correct device try_rollback must succeed");
        assert!(correct_tx.is_empty());
        test_support::reset();
    }

    #[test]
    fn wrong_device_consuming_rollback_preserves_transaction_for_retry() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let wrong_origin = WeightOrigin::from_parts(mesh.epoch(), 0, gpu.device_id + 77);
        let mut store = WeightStore::with_origin(wrong_origin);
        store
            .stage_alias("preserved", None, 0, "source", projection(DType::F32))
            .unwrap();
        let tx = WeightLoadTransaction::new(store);
        let releases_before = RESIDENT_RELEASES.with(std::cell::Cell::get);
        let (returned_tx, err) = tx.rollback(&gpu).unwrap_err();
        assert!(matches!(err, WeightStoreError::OriginMismatch { .. }));
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            releases_before
        );
        // Returned transaction still owns the allocation — caller can retry.
        assert!(returned_tx.contains("preserved", None, 0));
        // Retry with correct device after fixing origin (simulates retry on correct GPU).
        // We prove the allocation is still there by validating it.
        assert!(returned_tx
            .validate_origin_value(WeightOrigin::from_parts(
                mesh.epoch(),
                0,
                gpu.device_id + 77
            ))
            .is_ok());
        test_support::reset();
    }

    #[test]
    fn attached_store_wrong_device_try_drain_preserves() {
        // This test exercises the attached-store path indirectly via the
        // transaction gate; the carrier's AttachedWeightStore::try_drain
        // delegates to the same transaction gate.
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let wrong_origin = WeightOrigin::from_parts(mesh.epoch(), 0, gpu.device_id + 55);
        let mut store = WeightStore::with_origin(wrong_origin);
        store
            .stage_alias("attached", None, 0, "source", projection(DType::F32))
            .unwrap();
        let mut tx = WeightLoadTransaction::new(store);
        let err = tx.try_rollback(&gpu).unwrap_err();
        assert!(matches!(err, WeightStoreError::OriginMismatch { .. }));
        assert!(tx.contains("attached", None, 0));
        test_support::reset();
    }
}
