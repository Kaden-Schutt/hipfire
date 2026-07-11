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
//! **Scope.** Implemented placements: *whole-tensor upload* (single-GPU + all of
//! PP + every `Replicate`/`Pin`/`Tied`, and any sharding policy that degenerates
//! to a size-1 group); *expert-parallel `ExpertSharded`* on an `Ep>1` mesh (each
//! rank a compact blob of its owned experts — generic expert-outermost gather);
//! and dense tensor-parallel **`ColumnShard{axis:0}`** (PB-1a — contiguous
//! output-row split, format-agnostic) + **`RowShard`** (PB-1c — strided per-row
//! k-gather). Still returning a clear [`FulfillError`] at `Tp>1`: `FusedQkv` /
//! `HeadSharded` / `VocabShard` (and non-axis-0 `Column`) — the head-aware /
//! vocab gathers of PB-1b; refusing beats silently mis-placing.
//!
//! **Why a `source` closure, not `&HfqFile`.** A [`WeightEntry`] names tensors
//! *logically* (`"wq"`, `"ffn_down"`); the on-disk HFQ names are arch-specific
//! (prefix variants, GGUF `blk.N.*`). Reading them is the arch's knowledge, not
//! the engine's — so the caller passes a `source(entry) -> (raw bytes, dtype)`
//! closure (backed by its HFQ + name resolver), keeping the engine free of
//! on-disk naming. The dtype is the tensor's **real** on-disk quant type
//! (`Q4F16G64`/`MQ4`/`Q8_0`/`F16`/`F32`), so the placed tensor is forward-ready
//! (the right kernel dispatches), not an opaque `Raw` blob. This *pulls
//! complexity to the arch* and preserves the Tier-1 rule that the engine drives
//! placement without naming a device or an on-disk tensor. (The plan sketches
//! `fulfill_manifest(manifest, hfq, mesh)`; the source closure is the same shape
//! with the name-resolution seam made explicit.)

use crate::tp_shard::ShardConfig;
use crate::weight_manifest::{placement_devices, ShardPolicy, WeightEntry};
use hipfire_hardware::{DeviceMesh, DimKind};
use rdna_compute::{DType, GpuTensor};
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

    /// Move a placed handle out of the store (transferring ownership of its
    /// `GpuTensor`) — used when assembling an arch's weight struct *from* the
    /// store (the Phase-3 store→forward bridge). Leaves the cell empty.
    pub fn take(
        &mut self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<WeightHandle> {
        self.placements.remove(&(name.to_string(), layer, device))
    }

    /// Free every resident buffer (best-effort, on the device it was uploaded
    /// to) and consume the store — the transactional rollback for
    /// [`fulfill_manifest`], also the sharded-weight arm of `TpModel::free`.
    /// `Alias` handles own no buffer, so they are skipped.
    pub(crate) fn free_all(self, gpus: &crate::multi_gpu::Gpus) {
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
    // ExpertTensorSharded has its own fulfillment path, not a dense TP slice.
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

/// Slice a gate‖up expert blob `[2·inter, hidden]` for tensor-parallel rank `rank` of `tp`.
///
/// Layout: row-major, each row = `hidden/256` self-contained blocks of `block_bytes`.
/// Returns the paired slice `[2·(inter/tp), hidden]`: gate rows
/// `[rank·inter/tp .. (rank+1)·inter/tp)` followed immediately by up rows
/// `[inter + rank·inter/tp .. inter + (rank+1)·inter/tp)`.
/// Two contiguous byte-range copies — no dequant.
///
/// Errors if `inter % tp != 0` or `(inter/tp) % 256 != 0`.
pub fn expert_tp_column_pair(
    expert_blob: &[u8],
    inter: usize,
    hidden: usize,
    block_bytes: usize,
    rank: usize,
    tp: usize,
) -> Result<Vec<u8>, String> {
    if inter % tp != 0 {
        return Err(format!(
            "expert_tp_column_pair: inter {inter} not divisible by tp {tp}"
        ));
    }
    let slice = inter / tp;
    if slice % 256 != 0 {
        return Err(format!(
            "expert_tp_column_pair: inter/tp={slice} not divisible by group size 256"
        ));
    }
    let row_bytes = (hidden / 256) * block_bytes;
    let gate_start = rank * slice * row_bytes;
    let gate_end = gate_start + slice * row_bytes;
    let up_start = (inter + rank * slice) * row_bytes;
    let up_end = up_start + slice * row_bytes;
    let mut out = Vec::with_capacity(2 * slice * row_bytes);
    out.extend_from_slice(&expert_blob[gate_start..gate_end]);
    out.extend_from_slice(&expert_blob[up_start..up_end]);
    Ok(out)
}

/// Slice a down expert blob `[hidden, inter]` for tensor-parallel rank `rank` of `tp`.
///
/// Layout: row-major, each row = `inter/256` self-contained blocks of `block_bytes`.
/// Returns `[hidden, inter/tp]`: for each of `hidden` rows the block sub-range
/// `[rank·(inter/tp)/256 .. (rank+1)·(inter/tp)/256)`. Per-row strided gather —
/// no dequant.
///
/// Errors if `inter % tp != 0` or `(inter/tp) % 256 != 0`.
pub fn expert_tp_row_gather(
    expert_blob: &[u8],
    hidden: usize,
    inter: usize,
    block_bytes: usize,
    rank: usize,
    tp: usize,
) -> Result<Vec<u8>, String> {
    if inter % tp != 0 {
        return Err(format!(
            "expert_tp_row_gather: inter {inter} not divisible by tp {tp}"
        ));
    }
    let slice = inter / tp;
    if slice % 256 != 0 {
        return Err(format!(
            "expert_tp_row_gather: inter/tp={slice} not divisible by group size 256"
        ));
    }
    let row_bytes = (inter / 256) * block_bytes;
    let sub = (slice / 256) * block_bytes;
    let mut out = Vec::with_capacity(hidden * sub);
    for row in 0..hidden {
        let base = row * row_bytes + rank * sub;
        out.extend_from_slice(&expert_blob[base..base + sub]);
    }
    Ok(out)
}

/// Build the per-rank TP-sliced blob for an `ExpertTensorSharded` entry.
///
/// Iterates over all `n_experts`, extracts each expert's blob (`expert_bytes`
/// bytes at offset `e * expert_bytes`), and calls either
/// [`expert_tp_column_pair`] (ColumnShard inner — gate‖up split) or
/// [`expert_tp_row_gather`] (RowShard inner — down gather), then concatenates
/// the per-expert rank slices. Factored out of `fulfill_into` so the blob
/// construction (the correctness surface) is testable without a GPU.
pub fn build_expert_tp_blob(
    bytes: &[u8],
    n_experts: usize,
    expert_bytes: usize,
    inter: usize,
    hidden: usize,
    block_bytes: usize,
    rank: usize,
    tp: usize,
    inner: &ShardPolicy,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for e in 0..n_experts {
        let blob = &bytes[e * expert_bytes..(e + 1) * expert_bytes];
        let slice = match inner {
            ShardPolicy::ColumnShard { .. } => {
                expert_tp_column_pair(blob, inter, hidden, block_bytes, rank, tp)?
            }
            ShardPolicy::RowShard { .. } => {
                expert_tp_row_gather(blob, hidden, inter, block_bytes, rank, tp)?
            }
            other => {
                return Err(format!(
                    "build_expert_tp_blob: inner must be ColumnShard or RowShard, got {other:?}"
                ));
            }
        };
        out.extend_from_slice(&slice);
    }
    Ok(out)
}

/// Execute a weight manifest against a mesh: for each entry, compute its
/// placement (via the pure [`placement_devices`]) and upload the tensor's bytes
/// (from `source`) to every device it lands on, recording the result in a
/// [`WeightStore`]. This is the GPU counterpart of
/// [`crate::weight_manifest::plan_manifest`]'s weight-placement half.
///
/// `source(entry)` returns the **whole logical tensor's** raw bytes **and its
/// real on-disk dtype** (the caller resolves the on-disk name and reads its
/// HFQ). The tensor is uploaded under `entry.logical_shape` with that dtype, so
/// the placed tensor is forward-consumable (the correct kernel dispatches on the
/// quant type) — not an opaque `Raw` blob.
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
    F: Fn(&WeightEntry) -> Result<(Vec<u8>, DType), String>,
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
    F: Fn(&WeightEntry) -> Result<(Vec<u8>, DType), String>,
{
    debug_assert!(
        mesh.axes().iter().filter(|a| a.kind != DimKind::Pp).count() <= 1,
        "fulfill_manifest: single non-Pp axis only; composed Tp×Ep slicing is Phase 5b",
    );
    for entry in weights {
        let devices = placement_devices(entry, mesh, n_layers);
        let tp_axis = mesh.size_of(DimKind::Tp);
        let ep_axis = mesh.size_of(DimKind::Ep);

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
            if ep_axis > 1 {
                let tp_size = devices.len();
                let shard = ShardConfig::new(tp_size, false, *n_experts, *assign).map_err(|e| {
                    FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!("ExpertSharded: {e}"),
                    }
                })?;
                let (bytes, dtype) = source(entry).map_err(|e| FulfillError {
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
                    let mut tensor =
                        gpu.upload_raw(&compact, &shape).map_err(|e| FulfillError {
                            name: entry.name.clone(),
                            layer: entry.layer,
                            device: dev,
                            reason: format!("upload_raw failed: {e}"),
                        })?;
                    tensor.dtype = dtype;
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

        // Dense TP — ColumnShard on the OUTERMOST (row / output) axis is a clean
        // contiguous split (PB-1a): each row of a row-major quant blob is
        // independently quantized along k, so cutting the output-row dim into
        // `tp` equal parts is byte-clean for ANY quant format — no per-format
        // group math. Rank r stores only its `m/tp` rows: bytes [r·B/tp,(r+1)·B/tp).
        // (Row/FusedQkv/Head/Vocab, and non-axis-0 Column, still refuse below —
        // those need strided / head-aware / group-aligned gathers, PB-1b/1c.)
        if let ShardPolicy::ColumnShard { axis: 0 } = &entry.policy {
            if tp_axis > 1 {
                let tp = devices.len();
                let rows = *entry.logical_shape.first().unwrap_or(&0);
                if rows == 0 || rows % tp != 0 {
                    return Err(FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!(
                            "ColumnShard: outermost dim {rows} not divisible by Tp {tp}"
                        ),
                    });
                }
                let (bytes, dtype) = source(entry).map_err(|e| FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: *devices.first().unwrap_or(&0),
                    reason: format!("source read failed: {e}"),
                })?;
                if bytes.len() % tp != 0 {
                    return Err(FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!(
                            "ColumnShard: blob {} bytes not divisible by Tp {tp} \
                             (row-major quant rows must split evenly)",
                            bytes.len()
                        ),
                    });
                }
                let chunk = bytes.len() / tp;
                // Sharded logical shape: outermost dim becomes rows/tp.
                let mut shape = entry.logical_shape.clone();
                if let Some(first) = shape.first_mut() {
                    *first = rows / tp;
                }
                for (rank, &dev) in devices.iter().enumerate() {
                    let slice = &bytes[rank * chunk..(rank + 1) * chunk];
                    let gpu = gpus.devices.get(dev).ok_or_else(|| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("device {dev} out of range (have {})", gpus.devices.len()),
                    })?;
                    let mut tensor = gpu.upload_raw(slice, &shape).map_err(|e| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("upload_raw failed: {e}"),
                    })?;
                    tensor.dtype = dtype;
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

        // Dense TP — RowShard cuts the INNER (k / reduction) axis, so it is a
        // per-row STRIDED gather (PB-1c): rank r owns, of every one of the `m`
        // rows, the byte sub-range [r·rb/tp,(r+1)·rb/tp) where rb = row_bytes.
        // A row-major block-quant tensor stores each row as a run of contiguous
        // group-blocks, so this cut is quant-clean AS LONG AS rb/tp lands on a
        // group boundary — enforced upstream by `validate_manifest` (k %(tp·group)
        // == 0). Here we require the weaker byte-level `rb % tp == 0`; the
        // group-alignment guarantee is the manifest's. The gathered per-rank blob
        // is a valid row-major [m, k/tp] quant tensor the GEMV kernel consumes as-is.
        if let ShardPolicy::RowShard { .. } = &entry.policy {
            if tp_axis > 1 {
                let tp = devices.len();
                let rows = *entry.logical_shape.first().unwrap_or(&0);
                let inner: usize = entry.logical_shape.iter().skip(1).product();
                if rows == 0 || inner == 0 || inner % tp != 0 {
                    return Err(FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!("RowShard: inner dim {inner} not divisible by Tp {tp}"),
                    });
                }
                let (bytes, dtype) = source(entry).map_err(|e| FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: *devices.first().unwrap_or(&0),
                    reason: format!("source read failed: {e}"),
                })?;
                if bytes.len() % rows != 0 {
                    return Err(FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!(
                            "RowShard: blob {} bytes not a whole number of {rows} rows",
                            bytes.len()
                        ),
                    });
                }
                let row_bytes = bytes.len() / rows;
                if row_bytes % tp != 0 {
                    return Err(FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!(
                            "RowShard: row {row_bytes} bytes not divisible by Tp {tp} \
                             (k not group-aligned for this shard)"
                        ),
                    });
                }
                let sub = row_bytes / tp;
                // Sharded logical shape: the LAST dim (k) becomes k/tp.
                let mut shape = entry.logical_shape.clone();
                if let Some(last) = shape.last_mut() {
                    *last /= tp;
                }
                for (rank, &dev) in devices.iter().enumerate() {
                    // Gather rank r's k-slice out of every row.
                    let mut blob = Vec::with_capacity(rows * sub);
                    for row in 0..rows {
                        let base = row * row_bytes + rank * sub;
                        blob.extend_from_slice(&bytes[base..base + sub]);
                    }
                    let gpu = gpus.devices.get(dev).ok_or_else(|| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("device {dev} out of range (have {})", gpus.devices.len()),
                    })?;
                    let mut tensor = gpu.upload_raw(&blob, &shape).map_err(|e| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("upload_raw failed: {e}"),
                    })?;
                    tensor.dtype = dtype;
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

        // Expert-tensor-parallel (TP-of-experts): each rank in the Tp group holds
        // a TP-sliced fraction of every expert. For ColumnShard inner (gate‖up),
        // call `expert_tp_column_pair`; for RowShard inner (down), call
        // `expert_tp_row_gather`. Blob layout: [n_experts, ...] — expert-outermost.
        // Shape convention: [n_experts, 2*inter, hidden] for gate‖up (axis-1 = 2*inter),
        // [n_experts, hidden, inter] for down (axis-2 = inter).
        if let ShardPolicy::ExpertTensorSharded { n_experts, inner } = &entry.policy {
            if tp_axis > 1 {
                let tp = devices.len();
                let (bytes, dtype) = source(entry).map_err(|e| FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: *devices.first().unwrap_or(&0),
                    reason: format!("source read failed: {e}"),
                })?;
                // Derive block_bytes from dtype.
                let block_bytes: usize = match dtype {
                    DType::MQ2G256Lloyd => 72,
                    DType::MQ3G256Lloyd => 112,
                    _ => {
                        return Err(FulfillError {
                            name: entry.name.clone(),
                            layer: entry.layer,
                            device: *devices.first().unwrap_or(&0),
                            reason: format!(
                                "ExpertTensorSharded: unsupported dtype {dtype:?} \
                                 (expected MQ2G256Lloyd or MQ3G256Lloyd)"
                            ),
                        });
                    }
                };
                // Derive inter and hidden from logical_shape.
                // Gate‖up: [n_experts, 2*inter, hidden] → inter = shape[1]/2, hidden = shape[2]
                // Down:    [n_experts, hidden, inter]   → hidden = shape[1], inter = shape[2]
                let (inter, hidden) = match inner.as_ref() {
                    ShardPolicy::ColumnShard { .. } => {
                        let two_inter = entry.logical_shape.get(1).copied().unwrap_or(0);
                        let h = entry.logical_shape.get(2).copied().unwrap_or(0);
                        (two_inter / 2, h)
                    }
                    ShardPolicy::RowShard { .. } => {
                        let h = entry.logical_shape.get(1).copied().unwrap_or(0);
                        let i = entry.logical_shape.get(2).copied().unwrap_or(0);
                        (i, h)
                    }
                    _ => {
                        return Err(FulfillError {
                            name: entry.name.clone(),
                            layer: entry.layer,
                            device: *devices.first().unwrap_or(&0),
                            reason: format!(
                                "ExpertTensorSharded: inner must be ColumnShard or RowShard, \
                                 got {inner:?}"
                            ),
                        });
                    }
                };
                // Per-expert blob size (whole logical tensor / n_experts).
                if *n_experts == 0 || bytes.len() % n_experts != 0 {
                    return Err(FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: *devices.first().unwrap_or(&0),
                        reason: format!(
                            "ExpertTensorSharded: blob {} bytes not divisible by n_experts {}",
                            bytes.len(),
                            n_experts
                        ),
                    });
                }
                let expert_bytes = bytes.len() / n_experts;
                for (rank, &dev) in devices.iter().enumerate() {
                    // Build the per-rank blob: iterate over every expert,
                    // slice the per-expert blob for this rank, concatenate.
                    let per_rank_blob = build_expert_tp_blob(
                        &bytes,
                        *n_experts,
                        expert_bytes,
                        inter,
                        hidden,
                        block_bytes,
                        rank,
                        tp,
                        inner,
                    )
                    .map_err(|e| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: e,
                    })?;
                    let gpu = gpus.devices.get(dev).ok_or_else(|| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("device {dev} out of range (have {})", gpus.devices.len()),
                    })?;
                    let mut tensor = gpu
                        .upload_raw(&per_rank_blob, &entry.logical_shape)
                        .map_err(|e| FulfillError {
                            name: entry.name.clone(),
                            layer: entry.layer,
                            device: dev,
                            reason: format!("upload_raw failed: {e}"),
                        })?;
                    tensor.dtype = dtype;
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

        // Remaining dense TP slices across a real (≥2) group are not implemented
        // yet (PB-1b) — refuse rather than mis-place. A size-1 group degenerates
        // to a whole-tensor upload and is fine.
        if is_dense_tp_slice(&entry.policy) && tp_axis > 1 {
            return Err(FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: *devices.first().unwrap_or(&0),
                reason: format!(
                    "dense TP slicing (FusedQkv/Head/Vocab, or non-axis-0 Column) \
                     is not yet implemented (PB-1b); group size {} > 1",
                    devices.len()
                ),
            });
        }

        // Whole-tensor path: read once, upload the same bytes to each device.
        let (bytes, dtype) = source(entry).map_err(|e| FulfillError {
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
            let mut tensor =
                gpu.upload_raw(&bytes, &entry.logical_shape)
                    .map_err(|e| FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: dev,
                        reason: format!("upload_raw failed: {e}"),
                    })?;
            tensor.dtype = dtype;
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

    #[cfg(test)]
    mod tp_slice_tests {
        use super::*;
        // block_bytes=4 toy; inter=512 (2 groups of 256), hidden=256 (1 group), tp=2.
        // gate‖up blob: [2*inter=1024 rows, hidden=256] → 1 block/row → 1024 blocks × 4B.
        fn synth(nrows: usize, blocks_per_row: usize, bb: usize) -> Vec<u8> {
            (0..nrows * blocks_per_row)
                .flat_map(|b| (b as u32).to_le_bytes()[..bb].to_vec())
                .collect()
        }
        #[test]
        fn column_pair_takes_gate_then_up_halves() {
            let (inter, hidden, bb) = (512usize, 256usize, 4usize);
            let blob = synth(2 * inter, hidden / 256, bb); // 1024 rows, 1 block/row
            let r0 = expert_tp_column_pair(&blob, inter, hidden, bb, 0, 2).unwrap();
            // rank0 = gate rows [0..256) ++ up rows [512..768); 512 rows × 4B
            assert_eq!(r0.len(), 2 * (inter / 2) * (hidden / 256) * bb);
            assert_eq!(&r0[0..4], &0u32.to_le_bytes()); // gate row 0
            assert_eq!(&r0[256 * 4..256 * 4 + 4], &512u32.to_le_bytes()); // first up row = global row 512
        }
        #[test]
        fn row_gather_takes_group_subrange_per_row() {
            let (hidden, inter, bb) = (3usize, 512usize, 4usize);
            let blob = synth(hidden, inter / 256, bb); // 3 rows, 2 blocks/row
            let r1 = expert_tp_row_gather(&blob, hidden, inter, bb, 1, 2).unwrap();
            assert_eq!(r1.len(), hidden * (inter / 2 / 256) * bb); // 3 rows × 1 block × 4B
                                                                   // row 0's rank-1 block is global block index 1
            assert_eq!(&r1[0..4], &1u32.to_le_bytes());
            // row 1's rank-1 block is global block index 3
            assert_eq!(&r1[4..8], &3u32.to_le_bytes());
        }
        #[test]
        fn rejects_unaligned() {
            // (inter/tp) % 256 != 0 — both helpers
            assert!(expert_tp_column_pair(&[0u8; 16], 300, 256, 4, 0, 2).is_err());
            assert!(expert_tp_row_gather(&[0u8; 16], 3, 300, 4, 0, 2).is_err());
            // inter % tp != 0 (first guard) — both helpers
            assert!(expert_tp_column_pair(&[0u8; 16], 300, 256, 4, 0, 7).is_err());
            assert!(expert_tp_row_gather(&[0u8; 16], 3, 300, 4, 0, 7).is_err());
        }
    }

    #[test]
    fn expert_tensor_sharded_blob_construction() {
        // Synthetic 1-expert gate‖up blob for Tp-2:
        // inter=512, hidden=256, block_bytes=4 (toy).
        // Gate‖up blob shape: [2*inter=1024 rows, hidden/256=1 block/row] = 1024 × 4B.
        let (inter, hidden, bb) = (512usize, 256usize, 4usize);
        let n_experts = 1usize;
        // Build a single expert's gate‖up blob: 1024 rows × 1 block × 4B = 4096B.
        let expert_blob: Vec<u8> = (0u32..1024).flat_map(|i| i.to_le_bytes()).collect();
        assert_eq!(expert_blob.len(), 2 * inter * (hidden / 256) * bb);

        let inner_col = ShardPolicy::ColumnShard { axis: 0 };
        // rank 0 of tp=2 via column_pair helper directly:
        let expected_r0 = expert_tp_column_pair(&expert_blob, inter, hidden, bb, 0, 2).unwrap();
        let expected_r1 = expert_tp_column_pair(&expert_blob, inter, hidden, bb, 1, 2).unwrap();

        // build_expert_tp_blob for a 1-expert blob should equal expert_tp_column_pair directly.
        let got_r0 = build_expert_tp_blob(
            &expert_blob,
            n_experts,
            expert_blob.len(),
            inter,
            hidden,
            bb,
            0,
            2,
            &inner_col,
        )
        .unwrap();
        let got_r1 = build_expert_tp_blob(
            &expert_blob,
            n_experts,
            expert_blob.len(),
            inter,
            hidden,
            bb,
            1,
            2,
            &inner_col,
        )
        .unwrap();

        assert_eq!(got_r0.len(), expected_r0.len());
        assert_eq!(&got_r0[..4], &expected_r0[..4]);
        assert_eq!(got_r0, expected_r0);
        assert_eq!(got_r1, expected_r1);

        // Multi-expert (2): concatenation of per-expert slices.
        let two_expert_blob: Vec<u8> = (0u32..2048).flat_map(|i| i.to_le_bytes()).collect();
        let expert0 = &two_expert_blob[..expert_blob.len()];
        let expert1 = &two_expert_blob[expert_blob.len()..];
        let mut expected_multi = expert_tp_column_pair(expert0, inter, hidden, bb, 0, 2).unwrap();
        expected_multi.extend(expert_tp_column_pair(expert1, inter, hidden, bb, 0, 2).unwrap());

        let got_multi = build_expert_tp_blob(
            &two_expert_blob,
            2,
            expert_blob.len(),
            inter,
            hidden,
            bb,
            0,
            2,
            &inner_col,
        )
        .unwrap();
        assert_eq!(got_multi, expected_multi);

        // RowShard inner (down projection): hidden=3, inter=512, tp=2.
        let (h_down, i_down) = (3usize, 512usize);
        let down_blob: Vec<u8> = (0u32..(h_down * (i_down / 256)) as u32)
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let inner_row = ShardPolicy::RowShard { axis: 1 };
        let expected_down_r1 = expert_tp_row_gather(&down_blob, h_down, i_down, bb, 1, 2).unwrap();
        let got_down_r1 = build_expert_tp_blob(
            &down_blob,
            1,
            down_blob.len(),
            i_down,
            h_down,
            bb,
            1,
            2,
            &inner_row,
        )
        .unwrap();
        assert_eq!(got_down_r1, expected_down_r1);
    }

    // The dense-TP refusal path is checkable without a GPU: a row-shard on a
    // 2-device Tp mesh must Err before any upload. We can't build a real `Gpus`
    // without a GPU, so we assert the *decision* via placement + classifier
    // (the same predicates fulfill_manifest branches on).
    #[test]
    fn dense_tp_slice_would_refuse_on_multi_device() {
        // RowShard on a Tp-2 mesh: 2-device split → refusal decision. The refuse
        // predicate keys off the Tp axis size (not the device count).
        let tp2 = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let e = wl("wo", 0, ShardPolicy::RowShard { axis: 1 });
        let devs = placement_devices(&e, &tp2, 4);
        assert_eq!(devs.len(), 2);
        assert!(is_dense_tp_slice(&e.policy) && tp2.size_of(DimKind::Tp) > 1);
        // RowShard on an Ep-only mesh: placed across the whole EP group, but the
        // Tp axis is size 1 → NOT sliced/refused; it replicates (whole tensor per
        // rank) via the fall-through path. This is the EP-only fix.
        let ep2 = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        let redevs = placement_devices(&e, &ep2, 4);
        assert_eq!(redevs, vec![0, 1]);
        assert!(!(is_dense_tp_slice(&e.policy) && ep2.size_of(DimKind::Tp) > 1));
        // ExpertSharded on a 2-device Ep mesh places across the whole Ep group
        // and is sliced by expert (Ep axis > 1), never refused.
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
        assert!(!is_dense_tp_slice(&exp.policy) && ep2.size_of(DimKind::Ep) > 1);
        // Same dense entry on a single mesh degenerates to whole-tensor.
        let single = DeviceMesh::single();
        let devs1 = placement_devices(&e, &single, 4);
        assert_eq!(devs1, vec![0]);
        assert!(!(is_dense_tp_slice(&e.policy) && single.size_of(DimKind::Tp) > 1));
    }
}
