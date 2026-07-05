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
//! **Scope of this landing (deliberately minimal — "Simplicity First").** The
//! only placement a `Pp`/single mesh ever needs is *whole-tensor upload*:
//! pipeline parallelism bands whole layers across stages and never slices a
//! tensor; only tensor-parallel (`Column`/`Row`/`FusedQkv`/`Head`/`Vocab` at
//! `Tp>1`) and expert-parallel (`ExpertSharded`) actually cut a tensor, and the
//! plan defers live TP to Phase 5 and the arch-specific MoE packed-blob
//! convention to its own unit. So this driver handles the whole-tensor path
//! (which covers single-GPU + all of PP + every `Replicate`/`Pin`/`Tied`, and
//! any sharding policy that degenerates to a size-1 group) and returns a clear
//! [`FulfillError`] for the still-unimplemented slicing policies, rather than
//! speculatively re-encoding the quant-blob host-gather the bespoke loaders
//! already do.
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

/// True for a policy that cuts the tensor across a group of size ≥ 2 — the
/// slicing this landing defers. A group of size 1 (single-GPU / PP stage) never
/// slices, so a sharding policy there degenerates to a whole-tensor upload.
fn is_slicing_policy(policy: &ShardPolicy) -> bool {
    matches!(
        policy,
        ShardPolicy::ColumnShard { .. }
            | ShardPolicy::RowShard { .. }
            | ShardPolicy::FusedQkv { .. }
            | ShardPolicy::HeadSharded { .. }
            | ShardPolicy::VocabShard { .. }
            | ShardPolicy::ExpertSharded { .. }
    )
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
/// **Deferred policies return `Err`, not silent wrong placement:**
/// `ExpertSharded` (arch-specific MoE packed-blob + zeroed-dummy) and any dense
/// TP slice at `Tp>1` (`Column`/`Row`/`FusedQkv`/`Head`/`Vocab`) are Phase-5 /
/// EP-unit work; hitting one is a hard error so a caller can't mistake a
/// half-supported mesh for a full one.
///
/// Not transactional on mid-load OOM yet (matches the existing bespoke loaders,
/// which `hipMalloc` + leak): the §6 free-and-`Err`-all guard is a documented
/// follow-up. On the *first* failing cell it returns `Err` and drops the
/// partial store.
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

        // Slicing policies across a real (≥2) group are not implemented in this
        // landing — refuse rather than mis-place. A size-1 group degenerates to
        // a whole-tensor upload and is fine.
        if is_slicing_policy(&entry.policy) && devices.len() > 1 {
            let kind = match &entry.policy {
                ShardPolicy::ExpertSharded { .. } => {
                    "ExpertSharded (MoE packed-blob) upload is a separate EP unit"
                }
                _ => "dense TP slicing (Column/Row/FusedQkv/Head/Vocab) is Phase 5",
            };
            return Err(FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: *devices.first().unwrap_or(&0),
                reason: format!("{kind}; group size {} > 1", devices.len()),
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
    Ok(store)
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
    fn slicing_policy_classification() {
        assert!(is_slicing_policy(&ShardPolicy::RowShard { axis: 1 }));
        assert!(is_slicing_policy(&ShardPolicy::ColumnShard { axis: 0 }));
        assert!(is_slicing_policy(&ShardPolicy::ExpertSharded {
            n_experts: 8,
            assign: ExpertAssign::Stride
        }));
        assert!(is_slicing_policy(&ShardPolicy::VocabShard { axis: 0 }));
        // Whole-tensor policies never slice.
        assert!(!is_slicing_policy(&ShardPolicy::Replicate));
        assert!(!is_slicing_policy(&ShardPolicy::Pin(PinTarget::Embed)));
        assert!(!is_slicing_policy(&ShardPolicy::Tied {
            source: "x".into()
        }));
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

    // The refusal path is checkable without a GPU: a row-shard on a 2-device
    // Ep/Tp mesh must Err before any upload. We can't build a real `Gpus`
    // without a GPU, so we assert the *decision* via placement + classifier
    // (the same predicates fulfill_manifest branches on).
    #[test]
    fn deferred_slicing_would_refuse_on_multi_device() {
        // RowShard maps to the Tp group, so a Tp-2 mesh gives a 2-device split
        // → refusal. (On an Ep-only mesh it degenerates to a Tp singleton.)
        let tp2 = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let e = wl("wo", 0, ShardPolicy::RowShard { axis: 1 });
        let devs = placement_devices(&e, &tp2, 4);
        assert_eq!(devs.len(), 2);
        assert!(is_slicing_policy(&e.policy) && devs.len() > 1);
        // ExpertSharded refuses on a 2-device Ep mesh (its own axis).
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
        assert!(is_slicing_policy(&exp.policy) && edevs.len() > 1);
        // Same entry on a single mesh degenerates to whole-tensor (no refusal).
        let single = DeviceMesh::single();
        let devs1 = placement_devices(&e, &single, 4);
        assert_eq!(devs1, vec![0]);
        assert!(!(is_slicing_policy(&e.policy) && devs1.len() > 1));
    }
}
