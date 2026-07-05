// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `fulfill_manifest` — the GPU execution of a weight-placement plan (Phase 2
//! of the device-mesh plan, the "how" on top of the pure "where" in
//! [`crate::weight_manifest`]).
//!
//! [`crate::weight_manifest::plan_manifest`] already computes, deterministically
//! and on the CPU, *where every weight lands* (`placement = manifest × mesh`).
//! `fulfill_manifest` is the thin GPU driver that reads each tensor's bytes and
//! uploads them to the devices that plan names, returning a [`WeightStore`] —
//! the load-side *placement container* keyed by `(logical_name, layer, device)`, whose
//! value is a [`WeightHandle`] (`Resident` GPU tensor or `Alias` of another
//! entry). See docs/…/2026-07-05-device-mesh-transparent-parallelism.md §4.
//!
//! **Scope of this landing.** Two placements are implemented: *whole-tensor
//! upload* (single-GPU + all of PP + every `Replicate`/`Pin`/`Tied`, and any
//! sharding policy that degenerates to a size-1 group) and *expert-parallel
//! `ExpertSharded`* on an `Ep>1` mesh (each rank gets a compact blob of its
//! owned experts — a generic expert-outermost host gather). **Dense tensor-
//! parallel slicing** (`Column`/`Row`/`FusedQkv`/`Head`/`Vocab` at `Tp>1`)
//! returns a clear [`FulfillError`]: it needs the quant-blob row-gather that is
//! Phase-5 work, and refusing beats speculatively re-encoding it.
//!
//! **Why a `source` closure, not `&HfqFile`.** A [`WeightEntry`] names tensors
//! *logically* (`"wq"`, `"ffn_down"`); the on-disk HFQ names are arch-specific
//! (prefix variants, GGUF `blk.N.*`). Reading them is the arch's knowledge, not
//! the engine's — so the caller passes a `source(entry) -> raw bytes` closure
//! (backed by its HFQ + name resolver), keeping the engine free of on-disk
//! naming. This *pulls complexity to the arch* and preserves the Tier-1 rule
//! that the engine drives placement without naming a device or an on-disk
//! tensor. (The plan sketches `fulfill_manifest(manifest, hfq, mesh)`; the
//! source closure is the same shape with the name-resolution seam made explicit.)

use crate::tp_shard::ShardConfig;
use crate::weight_manifest::{placement_devices, ShardPolicy, WeightEntry};
use hipfire_hardware::DeviceMesh;
use rdna_compute::GpuTensor;
use std::collections::HashMap;

/// A placed weight: either a GPU-resident tensor or an alias to another entry
/// (tied embeddings / lm_head). Modelled as a handle enum so the deferred
/// `Paged(WeightId)` (weight-pager × mesh) slots in additively without
/// re-keying the store (device-mesh plan §4).
pub enum WeightHandle {
    /// The tensor's bytes live on the GPU (the device is the store key).
    Resident(GpuTensor),
    /// This entry reuses another entry's tensor (tied lm_head ↔ token_embd);
    /// the value is the source entry's logical name.
    Alias(String),
}

/// Load-side placement container, keyed by `(logical_name, layer, device_id)`.
/// Replaces the god-struct's placement bookkeeping: it records *where each
/// tensor landed*, independent of any arch's weight-struct shape. The `layer`
/// component is load-bearing — a per-layer weight shares one logical name
/// (`"wq"`) across every layer, so `(name, device)` alone would alias all
/// layers onto one cell (they all land on the same device under a PP stage).
/// This landing populates it; wiring the forward to read from it (instead of
/// arch fields) is Tier-2 / Phase 3 and deliberately out of scope here.
#[derive(Default)]
pub struct WeightStore {
    placements: HashMap<(String, Option<usize>, usize), WeightHandle>,
}

impl WeightStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of placed `(name, layer, device)` cells.
    pub fn len(&self) -> usize {
        self.placements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// The handle for `name` (of `layer`) on `device`, if placed there.
    pub fn get(&self, name: &str, layer: Option<usize>, device: usize) -> Option<&WeightHandle> {
        self.placements.get(&(name.to_string(), layer, device))
    }

    /// The devices a `(name, layer)` weight was placed on (ascending).
    pub fn devices_for(&self, name: &str, layer: Option<usize>) -> Vec<usize> {
        let mut ds: Vec<usize> = self
            .placements
            .keys()
            .filter(|(n, l, _)| n == name && *l == layer)
            .map(|(_, _, d)| *d)
            .collect();
        ds.sort_unstable();
        ds
    }

    fn insert(&mut self, name: &str, layer: Option<usize>, device: usize, handle: WeightHandle) {
        self.placements
            .insert((name.to_string(), layer, device), handle);
    }

    /// Free every resident buffer (best-effort, on the device it was uploaded
    /// to) and consume the store — the transactional rollback for
    /// [`fulfill_manifest`]. `Alias` handles own no buffer, so they are skipped.
    fn free_all(self, gpus: &crate::multi_gpu::Gpus) {
        for ((_, _, dev), handle) in self.placements {
            if let WeightHandle::Resident(t) = handle {
                if let Some(g) = gpus.devices.get(dev) {
                    let _ = g.hip.free(t.buf);
                }
            }
        }
    }
}

/// A weight that `fulfill_manifest` could not place. `device` is the cell it was
/// trying to reach (the `(coord)` of the plan's §4 `Err((coord, entry))`);
/// `reason` distinguishes a source-read failure, a GPU upload failure, or a
/// still-unimplemented slicing policy.
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

/// True for a **dense** TP policy that cuts a single matrix across a group of
/// size ≥ 2 — the row/column/head slicing this landing defers to Phase 5. A
/// group of size 1 (single-GPU / PP stage) never slices, so such a policy there
/// degenerates to a whole-tensor upload. `ExpertSharded` is *not* here: it is
/// handled directly (expert-outermost slicing is generic, unlike the quant-blob
/// row-gather the dense TP shards need).
fn is_dense_tp_slice(policy: &ShardPolicy) -> bool {
    matches!(
        policy,
        ShardPolicy::ColumnShard { .. }
            | ShardPolicy::RowShard { .. }
            | ShardPolicy::FusedQkv { .. }
            | ShardPolicy::HeadSharded { .. }
            | ShardPolicy::VocabShard { .. }
    )
}

/// Pack the bytes of a rank's *owned* experts into one compact blob. Experts are
/// the **outermost** dim of a routed-expert tensor (each expert is a
/// self-contained quant matrix), so per-expert byte ranges are contiguous and
/// the compaction is a generic host gather — no arch-specific quant knowledge.
/// (This is the *placement* the deepseek4 EP loader produces; the per-expert
/// pointer table + zeroed-dummy for non-owned experts is a forward-indexing
/// concern the arch owns, not part of where the bytes land.)
fn expert_compact_blob(bytes: &[u8], n_experts: usize, owned: &[usize]) -> Result<Vec<u8>, String> {
    if n_experts == 0 || bytes.len() % n_experts != 0 {
        return Err(format!(
            "experts blob {} not divisible by n_experts {n_experts}",
            bytes.len()
        ));
    }
    let per = bytes.len() / n_experts;
    let mut out = Vec::with_capacity(per * owned.len());
    for &e in owned {
        out.extend_from_slice(&bytes[e * per..(e + 1) * per]);
    }
    Ok(out)
}

/// Execute a weight manifest against a mesh: for each entry, compute its
/// placement (via the pure [`placement_devices`]) and upload the tensor's bytes
/// (from `source`) to every device it lands on, recording the result in a
/// [`WeightStore`]. This is the GPU counterpart of
/// [`crate::weight_manifest::plan_manifest`]'s weight-placement half.
///
/// `source(entry)` returns the **whole logical tensor's** raw bytes (the caller
/// resolves the on-disk name and reads its HFQ). The tensor is uploaded as a
/// `DType::Raw` blob under `entry.logical_shape` — a *placement* container, not
/// yet a forward-consumable operand (Phase 3 re-derives dtype from the manifest
/// when it wires the store into the forward).
///
/// `ExpertSharded` on an `Ep>1` mesh is handled directly: each rank receives a
/// compact blob of only its owned experts (the generic expert-outermost gather;
/// the arch's forward owns the per-expert pointer table + zeroed-dummy for
/// non-owned experts). **Dense TP slices at `Tp>1`** (`Column`/`Row`/`FusedQkv`/
/// `Head`/`Vocab`) still return `Err` — they need the quant-blob row-gather that
/// is Phase-5 work; refusing keeps a caller from mistaking a half-supported mesh
/// for a full one.
///
/// **Transactional** (device-mesh plan §6): on the first failing cell it frees
/// every already-uploaded buffer (best-effort, each on its own device) and
/// returns `Err` — never a half-loaded mesh leaking VRAM. Unlike the bespoke
/// loaders (which `hipMalloc` + leak on partial failure), a mid-load
/// source-read / shard-math / upload failure rolls back cleanly.
pub fn fulfill_manifest<F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpus: &crate::multi_gpu::Gpus,
    source: F,
) -> Result<WeightStore, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<Vec<u8>, String>,
{
    let mut store = WeightStore::new();
    match fulfill_into(&mut store, weights, mesh, n_layers, gpus, &source) {
        Ok(()) => Ok(store),
        Err(e) => {
            // Roll back every cell uploaded before the failure.
            store.free_all(gpus);
            Err(e)
        }
    }
}

/// The upload loop, writing into `store` so a partial result is reclaimable by
/// [`fulfill_manifest`]'s transactional rollback on error.
fn fulfill_into<F>(
    store: &mut WeightStore,
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpus: &crate::multi_gpu::Gpus,
    source: &F,
) -> Result<(), FulfillError>
where
    F: Fn(&WeightEntry) -> Result<Vec<u8>, String>,
{
    for entry in weights {
        let devices = placement_devices(entry, mesh, n_layers);

        // Tied: no upload — record an alias to the source entry on its device.
        if let ShardPolicy::Tied { source: src } = &entry.policy {
            let dev = devices.first().copied().unwrap_or(0);
            store.insert(
                &entry.name,
                entry.layer,
                dev,
                WeightHandle::Alias(src.clone()),
            );
            continue;
        }

        // Expert-parallel: each rank (device in the Ep group) gets a compact
        // blob of only its OWNED experts. Generic — expert-outermost slicing is
        // contiguous, no arch-specific quant handling. (Size-1 group falls
        // through to whole-tensor: all experts on the one device.)
        if let ShardPolicy::ExpertSharded { n_experts, assign } = &entry.policy {
            if devices.len() > 1 {
                let tp_size = devices.len();
                let shard = ShardConfig::new(tp_size, false, *n_experts, *assign).map_err(|e| {
                    FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!("ExpertSharded: {e}"),
                    }
                })?;
                let bytes = source(entry).map_err(|e| FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: *devices.first().unwrap_or(&0),
                    reason: format!("source read failed: {e}"),
                })?;
                for (rank, &dev) in devices.iter().enumerate() {
                    let owned = shard.experts_on_rank(rank);
                    let compact = expert_compact_blob(&bytes, *n_experts, &owned).map_err(|e| {
                        FulfillError {
                            name: entry.name.clone(),
                            layer: entry.layer,
                            device: dev,
                            reason: e,
                        }
                    })?;
                    // Compact shape: owned-expert count on the outermost dim.
                    let mut shape = entry.logical_shape.clone();
                    if let Some(first) = shape.first_mut() {
                        *first = owned.len();
                    }
                    let gpu = gpus.devices.get(dev).ok_or_else(|| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("device {dev} out of range (have {})", gpus.devices.len()),
                    })?;
                    let tensor = gpu.upload_raw(&compact, &shape).map_err(|e| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("upload_raw failed: {e}"),
                    })?;
                    store.insert(
                        &entry.name,
                        entry.layer,
                        dev,
                        WeightHandle::Resident(tensor),
                    );
                }
                continue;
            }
        }

        // Dense TP slicing across a real (≥2) group is not implemented in this
        // landing — refuse rather than mis-place. A size-1 group degenerates to
        // a whole-tensor upload and is fine.
        if is_dense_tp_slice(&entry.policy) && devices.len() > 1 {
            return Err(FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: *devices.first().unwrap_or(&0),
                reason: format!(
                    "dense TP slicing (Column/Row/FusedQkv/Head/Vocab) is Phase 5; \
                     group size {} > 1",
                    devices.len()
                ),
            });
        }

        // Whole-tensor path: read once, upload the same bytes to each device.
        let bytes = source(entry).map_err(|e| FulfillError {
            name: entry.name.clone(),
            layer: entry.layer,
            device: *devices.first().unwrap_or(&0),
            reason: format!("source read failed: {e}"),
        })?;
        for &dev in &devices {
            let gpu = gpus.devices.get(dev).ok_or_else(|| FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: dev,
                reason: format!("device {dev} out of range (have {})", gpus.devices.len()),
            })?;
            let tensor =
                gpu.upload_raw(&bytes, &entry.logical_shape)
                    .map_err(|e| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("upload_raw failed: {e}"),
                    })?;
            store.insert(
                &entry.name,
                entry.layer,
                dev,
                WeightHandle::Resident(tensor),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tp_shard::ExpertAssign;
    use crate::weight_manifest::PinTarget;
    use hipfire_hardware::DimKind;
    use rdna_compute::DType;

    fn wl(name: &str, layer: usize, policy: ShardPolicy) -> WeightEntry {
        WeightEntry::layer(name, layer, vec![8, 8], DType::F16, policy)
    }

    // These tests exercise the pure control-flow of fulfill_manifest that does
    // NOT need a GPU: policy classification and the deferred-slicing refusal.
    // The upload path is covered by the GPU example fulfill_manifest_probe.

    #[test]
    fn dense_tp_slice_classification() {
        assert!(is_dense_tp_slice(&ShardPolicy::RowShard { axis: 1 }));
        assert!(is_dense_tp_slice(&ShardPolicy::ColumnShard { axis: 0 }));
        assert!(is_dense_tp_slice(&ShardPolicy::VocabShard { axis: 0 }));
        // ExpertSharded is NOT a dense TP slice — it has its own generic path.
        assert!(!is_dense_tp_slice(&ShardPolicy::ExpertSharded {
            n_experts: 8,
            assign: ExpertAssign::Stride
        }));
        // Whole-tensor policies never slice.
        assert!(!is_dense_tp_slice(&ShardPolicy::Replicate));
        assert!(!is_dense_tp_slice(&ShardPolicy::Pin(PinTarget::Embed)));
        assert!(!is_dense_tp_slice(&ShardPolicy::Tied {
            source: "x".into()
        }));
    }

    #[test]
    fn expert_compact_blob_gathers_owned() {
        // 4 experts, 3 bytes each; rank owns experts [1, 3] (stride tp=2, rank 1).
        let bytes: Vec<u8> = (0..12).collect(); // e0=0..3 e1=3..6 e2=6..9 e3=9..12
        let owned = vec![1, 3];
        let out = expert_compact_blob(&bytes, 4, &owned).unwrap();
        assert_eq!(out, vec![3, 4, 5, 9, 10, 11]);
        // Non-divisible blob → error (shape/quant mismatch caught at load).
        assert!(expert_compact_blob(&bytes, 5, &owned).is_err());
        // Empty owned → empty blob (a rank owning no experts is caught upstream
        // by ShardConfig::new, but the gather itself is well-defined).
        assert_eq!(
            expert_compact_blob(&bytes, 4, &[]).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn store_keys_by_name_layer_and_device() {
        let mut s = WeightStore::new();
        // Same name on two devices, same layer → two cells.
        s.insert("wo", Some(0), 0, WeightHandle::Alias("src".into()));
        s.insert("wo", Some(0), 1, WeightHandle::Alias("src".into()));
        // Same name+device but a DIFFERENT layer → distinct cell (the bug the
        // byte-oracle caught: layer must be part of the key).
        s.insert("wo", Some(1), 0, WeightHandle::Alias("src".into()));
        assert_eq!(s.len(), 3);
        assert_eq!(s.devices_for("wo", Some(0)), vec![0, 1]);
        assert_eq!(s.devices_for("wo", Some(1)), vec![0]);
        assert!(matches!(
            s.get("wo", Some(0), 1),
            Some(WeightHandle::Alias(_))
        ));
        assert!(s.get("wo", Some(0), 2).is_none());
        assert!(s.get("wo", Some(2), 0).is_none());
    }

    // The dense-TP refusal path is checkable without a GPU: a row-shard on a
    // 2-device Tp mesh must Err before any upload. We can't build a real `Gpus`
    // without a GPU, so we assert the *decision* via placement + classifier
    // (the same predicates fulfill_manifest branches on).
    #[test]
    fn dense_tp_slice_would_refuse_on_multi_device() {
        // RowShard maps to the Tp group, so a Tp-2 mesh gives a 2-device split
        // → refusal. (On an Ep-only mesh it degenerates to a Tp singleton.)
        let tp2 = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let e = wl("wo", 0, ShardPolicy::RowShard { axis: 1 });
        let devs = placement_devices(&e, &tp2, 4);
        assert_eq!(devs.len(), 2);
        assert!(is_dense_tp_slice(&e.policy) && devs.len() > 1);
        // ExpertSharded on a 2-device Ep mesh is NOT refused — it has its own
        // path — but it does place across the whole Ep group.
        let ep2 = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let exp = wl(
            "experts",
            0,
            ShardPolicy::ExpertSharded {
                n_experts: 8,
                assign: ExpertAssign::Stride,
            },
        );
        let edevs = placement_devices(&exp, &ep2, 4);
        assert_eq!(edevs.len(), 2);
        assert!(!is_dense_tp_slice(&exp.policy));
        // Same dense entry on a single mesh degenerates to whole-tensor (no refusal).
        let single = DeviceMesh::single();
        let devs1 = placement_devices(&e, &single, 4);
        assert_eq!(devs1, vec![0]);
        assert!(!(is_dense_tp_slice(&e.policy) && devs1.len() > 1));
    }
}
