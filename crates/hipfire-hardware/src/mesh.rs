// SPDX-License-Identifier: MIT
//! `DeviceMesh` — the single abstraction that describes how logical devices are
//! arranged for pipeline / tensor / expert parallelism, consumed identically by
//! the forward executor (compute placement) and the loader (weight/state
//! placement). See docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md.
//!
//! **Named-axis semantics are primary.** A mesh is an ordered set of typed axes
//! `{Pp, Tp, Ep}`; a device's *coordinate* is an index tuple over the axes, and
//! its *collective group along an axis* is every device sharing the other
//! coordinates. This already expresses coexisting TP and EP at different group
//! sizes (a MoE model reduces its expert op over the `Ep` group and attention
//! over the `Tp` group) — exactly how Megatron/JAX meshes work.
//!
//! The common **rectangular** case (uniform size per axis) is implemented here
//! and built via [`DeviceMesh::rect`]. Degenerate cases: single-GPU = no axes
//! (`rect(&[])`, one device); PP-only = `[{Pp, N}]`; EP-only = `[{Ep, N}]`.
//!
//! **Raggedness (a `Dimension` tree — different Tp/Ep size per Pp stage, for
//! heterogeneous/mixed-arch fleets) is a Phase-5b extension**, not built here.
//! The rectangular mesh is the "all sub-trees identical" special case, so the
//! tree is a future superset, not a rewrite.

use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonically-increasing epoch counter for identity-sensitive comparison.
/// Each call to [`DeviceMesh::rect`] or [`DeviceMesh::single`] bumps the
/// counter and assigns the new value as the mesh's epoch.
static NEXT_MESH_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Allocate the next [`MeshEpoch`] from the global counter.
///
/// Uses a CAS loop (not a blind `fetch_add`) so that we never advance past
/// `u64::MAX`.  The value `u64::MAX` is reserved as exhaustion sentinel and is
/// never issued; only values through `u64::MAX - 1` are issuable.  When the
/// counter reaches the sentinel the function panics — there is no recovery path
/// because reissuing epochs would silently break identity-sensitive equality.
/// The check happens *before* the CAS, so even if a panic is caught the global
/// state is never left wrapping around.
fn next_epoch() -> MeshEpoch {
    let mut current = NEXT_MESH_EPOCH.load(Ordering::Relaxed);
    loop {
        if current == u64::MAX {
            panic!("MeshEpoch exhausted: no remaining issuable epochs");
        }
        match NEXT_MESH_EPOCH.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return MeshEpoch(current),
            Err(actual) => current = actual,
        }
    }
}

/// Opaque epoch identifier that distinguishes independently-constructed meshes.
///
/// Two [`DeviceMesh`] values compare equal only when they share the same epoch
/// (i.e., one was derived from the other via [`Clone`] or
/// [`DeviceMesh::squeezed`]).  The tuple field is private; the type itself is
/// public.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MeshEpoch(u64);

/// The parallelism axes a device coordinate can range over.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DimKind {
    /// Pipeline-parallel: layers are banded across stages; the residual is
    /// handed stage→stage (`BandXfer`). Point-to-point, never all-reduced.
    Pp,
    /// Tensor-parallel: dense GEMMs (attention/MLP) sharded within a group,
    /// all-reduced after the row-sharded op.
    Tp,
    /// Expert-parallel: MoE experts sharded within a group, all-reduced after
    /// the expert op.
    Ep,
}

/// A cross-device sync the executor injects for an op, derived (not
/// hand-written) from the op's weight [`ShardPolicy`] — the "single source of
/// truth" partitioner: a row-sharded / expert-sharded weight *mechanically
/// implies* an all-reduce over its axis group; a pipeline-band boundary implies
/// a residual copy. See docs/…/device-mesh §1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollectiveHint {
    /// All-reduce the op's output over the device group along `kind`
    /// (`Tp` for row-sharded dense, `Ep` for MoE experts). The reduce
    /// element count is applied at execution (`n_rows * dim`).
    AllReduce { kind: DimKind },
    /// Copy the residual stream across a pipeline-stage boundary (PP), from
    /// global device `src` to `dst`.
    BandXfer { src: usize, dst: usize },
}

/// One rectangular axis of the mesh.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Axis {
    pub kind: DimKind,
    pub size: usize,
}

/// A rectangular device mesh: an ordered list of typed axes. A global device id
/// is the row-major flattening of a coordinate tuple over the axes (last axis
/// varies fastest). An empty axis list is the single-device (1×1) mesh.
///
/// Equality is **identity-sensitive**: two meshes compare equal only when they
/// share the same [`MeshEpoch`] — i.e., one was derived from the other via
/// [`Clone`] or [`DeviceMesh::squeezed`].  Independently constructed meshes
/// that happen to have the same shape are *not* equal.
#[derive(Clone, Debug)]
pub struct DeviceMesh {
    axes: Vec<Axis>,
    epoch: MeshEpoch,
}

impl PartialEq for DeviceMesh {
    fn eq(&self, other: &Self) -> bool {
        self.epoch == other.epoch
    }
}

impl Eq for DeviceMesh {}

impl DeviceMesh {
    /// Build a rectangular mesh from `(kind, size)` pairs. Sizes must be ≥ 1;
    /// size-1 axes are kept (so `group_along` over them is a singleton) — call
    /// [`DeviceMesh::squeezed`] to drop them if you want a minimal shape.
    ///
    /// Each call issues a fresh [`MeshEpoch`] so that independently constructed
    /// meshes never compare equal, even when their axes are identical.
    pub fn rect(axes: &[(DimKind, usize)]) -> Self {
        let axes = axes
            .iter()
            .map(|&(kind, size)| Axis {
                kind,
                size: size.max(1),
            })
            .collect();
        Self {
            axes,
            epoch: next_epoch(),
        }
    }

    /// The single-device (1×1) mesh — no axes, exactly one device. This is what
    /// the unified `run_layer_program` runs the single-GPU path as.
    ///
    /// Each call issues a fresh [`MeshEpoch`] (same identity-sensitivity rule
    /// as [`rect`](Self::rect)).
    pub fn single() -> Self {
        Self {
            axes: Vec::new(),
            epoch: next_epoch(),
        }
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// The opaque epoch that distinguishes this mesh instance from
    /// independently-constructed meshes — even those with identical topology.
    pub fn epoch(&self) -> MeshEpoch {
        self.epoch
    }

    /// Total number of logical devices = product of axis sizes (1 for a mesh
    /// with no axes).
    pub fn n_devices(&self) -> usize {
        self.axes.iter().map(|a| a.size).product::<usize>().max(1)
    }

    /// The size of the first axis of the given kind (1 if absent).
    pub fn size_of(&self, kind: DimKind) -> usize {
        self.axes
            .iter()
            .find(|a| a.kind == kind)
            .map_or(1, |a| a.size)
    }

    /// Whether this mesh has more than one device along `kind`.
    pub fn has_axis(&self, kind: DimKind) -> bool {
        self.axes.iter().any(|a| a.kind == kind && a.size > 1)
    }

    /// Coordinate tuple (one index per axis) for a global device id.
    pub fn coord_of(&self, dev: usize) -> Vec<usize> {
        let mut rem = dev;
        let mut coord = vec![0usize; self.axes.len()];
        // Row-major: last axis varies fastest.
        for i in (0..self.axes.len()).rev() {
            let sz = self.axes[i].size;
            coord[i] = rem % sz;
            rem /= sz;
        }
        coord
    }

    /// Global device id for a coordinate tuple (inverse of [`coord_of`]).
    pub fn device_of(&self, coord: &[usize]) -> usize {
        debug_assert_eq!(coord.len(), self.axes.len());
        let mut id = 0usize;
        for (i, a) in self.axes.iter().enumerate() {
            id = id * a.size + coord[i].min(a.size - 1);
        }
        id
    }

    /// The collective group along `kind` for the device at `coord`: every device
    /// sharing all *other* coordinates, ordered by their index along `kind`.
    /// Returns the ids as a `Vec<usize>` suitable for
    /// `Gpus::all_reduce_sum_f32[_peer](&group, …)`. If the mesh has no axis of
    /// `kind`, the group is just this device (singleton) — the all-reduce is
    /// then the identity, matching the single-GPU / no-op case.
    pub fn group_along(&self, kind: DimKind, coord: &[usize]) -> Vec<usize> {
        let Some(axis_idx) = self.axes.iter().position(|a| a.kind == kind) else {
            return vec![self.device_of(coord)];
        };
        let size = self.axes[axis_idx].size;
        (0..size)
            .map(|k| {
                let mut c = coord.to_vec();
                c[axis_idx] = k;
                self.device_of(&c)
            })
            .collect()
    }

    /// Which pipeline stage (`Pp` coordinate) owns layer `layer` of `n_layers`,
    /// using a uniform band split (max−min ≤ 1 layer per stage, earlier stages
    /// take the remainder). No `Pp` axis → stage 0.
    pub fn stage_for_layer(&self, layer: usize, n_layers: usize) -> usize {
        let p = self.size_of(DimKind::Pp);
        if p <= 1 || n_layers == 0 {
            return 0;
        }
        let base = n_layers / p;
        let rem = n_layers % p;
        let mut start = 0usize;
        for s in 0..p {
            let cnt = base + if s < rem { 1 } else { 0 };
            if layer < start + cnt {
                return s;
            }
            start += cnt;
        }
        p - 1
    }

    /// The residual-stream `BandXfer` to inject *after* `layer`, if the next
    /// layer lives on a different pipeline stage. `None` at the last layer or
    /// when there is no `Pp` axis. NOTE: for a pure `Pp` (P×1) mesh the device
    /// id equals the stage; composed meshes (per-stage tp-rank mapping) are
    /// Phase 5b — asserted single-axis here.
    pub fn band_xfer_after(&self, layer: usize, n_layers: usize) -> Option<CollectiveHint> {
        if !self.has_axis(DimKind::Pp) || layer + 1 >= n_layers {
            return None;
        }
        debug_assert_eq!(
            self.axes.len(),
            1,
            "band_xfer_after: pure Pp mesh only; composed meshes are Phase 5b",
        );
        let s = self.stage_for_layer(layer, n_layers);
        let s1 = self.stage_for_layer(layer + 1, n_layers);
        (s != s1).then_some(CollectiveHint::BandXfer { src: s, dst: s1 })
    }

    /// Drop size-1 axes, yielding the minimal equivalent shape.
    /// The returned mesh retains the same [`MeshEpoch`] — topology normalization
    /// is the same mesh identity.
    pub fn squeezed(&self) -> Self {
        Self {
            axes: self.axes.iter().copied().filter(|a| a.size > 1).collect(),
            epoch: self.epoch,
        }
    }

    /// Every device in the compute grid of the pipeline stage at `coord`: all
    /// devices sharing `coord`'s `Pp` index, varying every non-`Pp` axis (Tp × Ep).
    /// This is the placement set for a stage's weights/state — replicated weights
    /// land on all of them; Tp/Ep-sharded weights are sliced along their axis and
    /// replicated across the other. Degenerate: no non-`Pp` axis → the stage's
    /// single device; no `Pp` axis → the whole mesh. Correct for composed meshes
    /// (pure topology). Ordering is ascending device id, so on a single-axis
    /// non-`Pp` mesh the index into the returned `Vec` equals the shard rank.
    pub fn stage_devices(&self, coord: &[usize]) -> Vec<usize> {
        let pp_idx = self.axes.iter().position(|a| a.kind == DimKind::Pp);
        (0..self.n_devices())
            .filter(|&d| pp_idx.is_none_or(|i| self.coord_of(d)[i] == coord[i]))
            .collect()
    }
}

impl Default for DeviceMesh {
    fn default() -> Self {
        Self::single()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_clone_preserved() {
        let m = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        assert_eq!(m.epoch(), m.clone().epoch());
        // Clone preserves PartialEq equality (same epoch).
        assert_eq!(m, m.clone());
    }

    #[test]
    fn epoch_rect_independent_instances_distinct() {
        let a = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let b = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        assert_ne!(a.epoch(), b.epoch());
        assert_ne!(a, b);
    }

    #[test]
    fn epoch_single_independent_instances_distinct() {
        let a = DeviceMesh::single();
        let b = DeviceMesh::single();
        assert_ne!(a.epoch(), b.epoch());
        assert_ne!(a, b);
    }

    #[test]
    fn epoch_squeezed_preserves_epoch_and_equality() {
        let m = DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 1)]);
        let sq = m.squeezed();
        assert_eq!(m.epoch(), sq.epoch());
        assert_eq!(m, sq);
        // Squeezed axes are minimal; verify shape was normalized.
        assert_eq!(
            sq.axes(),
            &[Axis {
                kind: DimKind::Tp,
                size: 2
            }]
        );
    }

    #[test]
    fn single_is_one_device_no_axes() {
        let m = DeviceMesh::single();
        assert_eq!(m.n_devices(), 1);
        assert_eq!(m.coord_of(0), Vec::<usize>::new());
        // No Tp axis → all-reduce group is the singleton {0} (identity).
        assert_eq!(m.group_along(DimKind::Tp, &[]), vec![0]);
    }

    #[test]
    fn pp_only_n_by_1() {
        let m = DeviceMesh::rect(&[(DimKind::Pp, 4)]);
        assert_eq!(m.n_devices(), 4);
        assert_eq!(m.coord_of(2), vec![2]);
        assert_eq!(m.device_of(&[3]), 3);
        // No Tp/Ep axis → singleton groups (PP never all-reduces).
        assert_eq!(m.group_along(DimKind::Tp, &[2]), vec![2]);
    }

    #[test]
    fn ep_only_1_by_n_group_is_all_devices() {
        let m = DeviceMesh::rect(&[(DimKind::Ep, 4)]);
        assert_eq!(m.n_devices(), 4);
        // Ep group for any device = all 4 (they share the empty "other" coords).
        assert_eq!(m.group_along(DimKind::Ep, &[1]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn stacked_2x2_pp_tp_coords_and_groups() {
        // axes = [Pp:2, Tp:2]; row-major, Tp varies fastest.
        let m = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)]);
        assert_eq!(m.n_devices(), 4);
        // device 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
        assert_eq!(m.coord_of(0), vec![0, 0]);
        assert_eq!(m.coord_of(1), vec![0, 1]);
        assert_eq!(m.coord_of(3), vec![1, 1]);
        assert_eq!(m.device_of(&[1, 0]), 2);
        // Tp group of device 2=(1,0): share Pp=1, vary Tp → devices 2,3.
        assert_eq!(m.group_along(DimKind::Tp, &[1, 0]), vec![2, 3]);
        // Tp group of device 1=(0,1): share Pp=0 → devices 0,1.
        assert_eq!(m.group_along(DimKind::Tp, &[0, 1]), vec![0, 1]);
        // Pp "group" (not all-reduced, but the accessor is symmetric) of
        // device 1=(0,1): share Tp=1, vary Pp → devices 1,3.
        assert_eq!(m.group_along(DimKind::Pp, &[0, 1]), vec![1, 3]);
    }

    #[test]
    fn stage_for_layer_uniform_band_split() {
        // 4 stages, 10 layers → counts 3,3,2,2 (earlier stages take remainder).
        let m = DeviceMesh::rect(&[(DimKind::Pp, 4)]);
        let stages: Vec<usize> = (0..10).map(|l| m.stage_for_layer(l, 10)).collect();
        assert_eq!(stages, vec![0, 0, 0, 1, 1, 1, 2, 2, 3, 3]);
        // No Pp axis → always stage 0.
        let ep = DeviceMesh::rect(&[(DimKind::Ep, 4)]);
        assert_eq!(ep.stage_for_layer(5, 10), 0);
    }

    #[test]
    fn band_xfer_at_stage_boundaries_only() {
        let m = DeviceMesh::rect(&[(DimKind::Pp, 2)]); // 2 stages, 4 layers → 2,2
                                                       // boundary between layer 1 (stage 0) and layer 2 (stage 1).
        assert_eq!(m.band_xfer_after(0, 4), None);
        assert_eq!(
            m.band_xfer_after(1, 4),
            Some(CollectiveHint::BandXfer { src: 0, dst: 1 })
        );
        assert_eq!(m.band_xfer_after(2, 4), None);
        assert_eq!(m.band_xfer_after(3, 4), None); // last layer
    }

    #[test]
    fn coord_roundtrip_all_devices() {
        let m = DeviceMesh::rect(&[(DimKind::Pp, 3), (DimKind::Ep, 2)]);
        for d in 0..m.n_devices() {
            assert_eq!(m.device_of(&m.coord_of(d)), d);
        }
    }

    #[test]
    fn stage_devices_spans_stage_grid() {
        // single: exactly one device.
        assert_eq!(DeviceMesh::single().stage_devices(&[]), vec![0]);
        // Pp-only: the stage's single device (the coord's Pp index).
        let pp = DeviceMesh::rect(&[(DimKind::Pp, 3)]);
        assert_eq!(pp.stage_devices(&[1]), vec![1]);
        // Ep-only: the whole EP group (every rank runs full attention).
        let ep = DeviceMesh::rect(&[(DimKind::Ep, 4)]);
        assert_eq!(ep.stage_devices(&[0]), vec![0, 1, 2, 3]);
        // Tp-only: the whole TP group.
        let tp = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        assert_eq!(tp.stage_devices(&[0]), vec![0, 1]);
        // Composed Pp×Tp: stage 1 = the Tp group at Pp=1 → devices 2,3.
        let pptp = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)]);
        assert_eq!(pptp.stage_devices(&[1, 0]), vec![2, 3]);
        // Composed Tp×Ep (no Pp): the full sub-grid.
        let tpep = DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 2)]);
        assert_eq!(tpep.stage_devices(&[0, 0]), vec![0, 1, 2, 3]);
    }
}
