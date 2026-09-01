// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure rectangular device topology for pipeline, tensor, and expert
//! parallelism.
//!
//! A mesh is an ordered set of named axes. Device IDs are the row-major
//! flattening of coordinates (the final axis varies fastest). This module is
//! deliberately independent of GPU handles, carrier policy, loading, and
//! allocation: it only answers placement and collective-group questions.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MESH_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Identity of one admitted mesh generation.
///
/// Cloning or squeezing a mesh preserves this identity. A fresh call to
/// [`DeviceMesh::single`] or [`DeviceMesh::rect`] receives a new epoch, even
/// when its shape happens to match another mesh.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MeshEpoch(u64);

impl MeshEpoch {
    /// Return the process-local epoch number for diagnostics and cache keys.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// The named parallelism dimensions of a device coordinate.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DimKind {
    /// Pipeline stages. Layers are banded across this axis and residuals cross
    /// stage boundaries point-to-point; PP is never an all-reduce axis.
    Pp,
    /// Tensor-parallel ranks participating in dense row-sharded collectives.
    Tp,
    /// Expert-parallel ranks participating in routed-expert collectives.
    Ep,
}

/// One axis in a rectangular mesh.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Axis {
    pub kind: DimKind,
    pub size: usize,
}

/// A collective or point-to-point operation implied by mesh placement.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CollectiveHint {
    /// Reduce values across the named axis group.
    AllReduce { kind: DimKind },
    /// Transfer the residual from one pipeline stage to the next.
    ///
    /// `src` and `dst` are stage coordinates (not physical device IDs). Use
    /// [`DeviceMesh::stage_devices`] to expand a stage coordinate into its
    /// global device IDs when TP or EP is composed with PP.
    BandXfer { src: usize, dst: usize },
}

/// A rectangular topology cannot represent a cardinality larger than the
/// platform's `usize` range.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MeshError {
    CardinalityOverflow,
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CardinalityOverflow => {
                f.write_str("rectangular device mesh cardinality overflow")
            }
        }
    }
}

impl std::error::Error for MeshError {}

/// A rectangular named-axis mesh.
///
/// The empty axis list is the single-device topology. Size-one axes remain in
/// the shape supplied to [`DeviceMesh::rect`]; [`DeviceMesh::squeezed`] drops
/// those axes while retaining the same [`MeshEpoch`]. Equality is
/// identity-sensitive: independently constructed meshes compare unequal even
/// when their shapes match.
#[derive(Clone, Debug)]
pub struct DeviceMesh {
    axes: Vec<Axis>,
    n_devices: usize,
    epoch: MeshEpoch,
}

impl PartialEq for DeviceMesh {
    fn eq(&self, other: &Self) -> bool {
        // MeshEpoch is intentionally identity-sensitive: independently
        // constructed meshes are distinct even when their shapes match.
        self.epoch == other.epoch
    }
}

impl Eq for DeviceMesh {}

impl DeviceMesh {
    /// Build a rectangular mesh from `(kind, size)` pairs.
    ///
    /// Axis sizes are normalized to at least one so every mesh has a valid
    /// coordinate space and at least one logical device. Named axes are kept
    /// in caller order; the final axis varies fastest in the flattened ID.
    ///
    /// The cardinality is checked while constructing the mesh. Callers must
    /// handle [`MeshError::CardinalityOverflow`] instead of observing a
    /// wrapped device count.
    pub fn rect(axes: &[(DimKind, usize)]) -> Result<Self, MeshError> {
        let mut normalized = Vec::with_capacity(axes.len());
        let mut n_devices = 1usize;
        for &(kind, size) in axes {
            let size = size.max(1);
            n_devices = n_devices
                .checked_mul(size)
                .ok_or(MeshError::CardinalityOverflow)?;
            normalized.push(Axis { kind, size });
        }
        Ok(Self {
            axes: normalized,
            n_devices,
            epoch: fresh_epoch(),
        })
    }

    /// The single-device topology: one logical device and no named axes.
    pub fn single() -> Result<Self, MeshError> {
        Self::rect(&[])
    }

    /// Ordered named axes of this mesh.
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// Identity of this mesh generation.
    pub fn epoch(&self) -> MeshEpoch {
        self.epoch
    }

    /// Total number of logical devices (one for an empty mesh).
    pub fn n_devices(&self) -> usize {
        self.n_devices
    }

    /// Size of the first axis with `kind`, or one when it is absent.
    pub fn size_of(&self, kind: DimKind) -> usize {
        self.axes
            .iter()
            .find(|axis| axis.kind == kind)
            .map_or(1, |axis| axis.size)
    }

    /// Whether this mesh has a non-degenerate axis of `kind`.
    pub fn has_axis(&self, kind: DimKind) -> bool {
        self.axes
            .iter()
            .any(|axis| axis.kind == kind && axis.size > 1)
    }

    /// Convert a flattened row-major device ID to an axis coordinate.
    pub fn coord_of(&self, dev: usize) -> Vec<usize> {
        let mut rem = dev % self.n_devices();
        let mut coord = vec![0; self.axes.len()];
        for (index, axis) in self.axes.iter().enumerate().rev() {
            coord[index] = rem % axis.size;
            rem /= axis.size;
        }
        coord
    }

    /// Convert an axis coordinate to a flattened row-major device ID.
    ///
    /// Coordinates are debug-asserted to have the mesh rank. In non-debug
    /// builds, missing entries default to zero and out-of-range entries clamp
    /// to the final valid index, preserving a total function for diagnostics.
    pub fn device_of(&self, coord: &[usize]) -> usize {
        debug_assert_eq!(coord.len(), self.axes.len());
        self.axes.iter().enumerate().fold(0, |id, (index, axis)| {
            let value = coord.get(index).copied().unwrap_or(0);
            id * axis.size + value.min(axis.size - 1)
        })
    }

    /// Return the devices sharing all coordinates except `kind`, ordered by
    /// their index along that axis. An absent axis yields this device alone.
    pub fn group_along(&self, kind: DimKind, coord: &[usize]) -> Vec<usize> {
        let Some(axis_index) = self.axes.iter().position(|axis| axis.kind == kind) else {
            return vec![self.device_of(coord)];
        };
        let size = self.axes[axis_index].size;
        let mut base = self.normalized_coord(coord);
        (0..size)
            .map(|index| {
                base[axis_index] = index;
                self.device_of(&base)
            })
            .collect()
    }

    /// Return the pipeline stage containing `layer` under a uniform banding.
    ///
    /// The first `n_layers % pp_size` stages receive one extra layer. A mesh
    /// without a PP axis has one stage. Valid layer indexes always map to a
    /// non-empty stage; an out-of-range index is clamped to the final stage.
    pub fn stage_for_layer(&self, layer: usize, n_layers: usize) -> usize {
        let stages = self.size_of(DimKind::Pp);
        if stages <= 1 || n_layers == 0 {
            return 0;
        }
        let base = n_layers / stages;
        let remainder = n_layers % stages;
        let mut start = 0;
        for stage in 0..stages {
            let count = base + usize::from(stage < remainder);
            if layer < start + count {
                return stage;
            }
            start += count;
        }
        stages - 1
    }

    /// Return a point-to-point hint when the next layer crosses a PP band.
    pub fn band_xfer_after(&self, layer: usize, n_layers: usize) -> Option<CollectiveHint> {
        if n_layers == 0 || layer >= n_layers.saturating_sub(1) {
            return None;
        }
        let src = self.stage_for_layer(layer, n_layers);
        let dst = self.stage_for_layer(layer + 1, n_layers);
        (src != dst).then_some(CollectiveHint::BandXfer { src, dst })
    }

    /// Expand the PP stage in `coord` to its global device IDs.
    ///
    /// For a mesh without PP, all devices belong to the sole stage. For a
    /// composed mesh, every TP/EP coordinate in the selected PP stage is
    /// returned in row-major order.
    pub fn stage_devices(&self, coord: &[usize]) -> Vec<usize> {
        let Some(pp_index) = self.axes.iter().position(|axis| axis.kind == DimKind::Pp) else {
            return (0..self.n_devices()).collect();
        };
        let stage = coord
            .get(pp_index)
            .copied()
            .unwrap_or(0)
            .min(self.axes[pp_index].size - 1);
        (0..self.n_devices())
            .filter(|&device| self.coord_of(device)[pp_index] == stage)
            .collect()
    }

    /// Drop size-one axes while preserving the mesh epoch identity.
    pub fn squeezed(&self) -> Self {
        Self {
            axes: self
                .axes
                .iter()
                .copied()
                .filter(|axis| axis.size > 1)
                .collect(),
            n_devices: self.n_devices,
            epoch: self.epoch,
        }
    }

    fn normalized_coord(&self, coord: &[usize]) -> Vec<usize> {
        debug_assert_eq!(coord.len(), self.axes.len());
        self.axes
            .iter()
            .enumerate()
            .map(|(index, axis)| coord.get(index).copied().unwrap_or(0).min(axis.size - 1))
            .collect()
    }
}

impl Default for DeviceMesh {
    fn default() -> Self {
        Self::single().expect("single-device mesh cannot overflow")
    }
}

fn fresh_epoch() -> MeshEpoch {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_is_one_device_and_identity_collectives_are_noops() {
        let mesh = DeviceMesh::single().unwrap();
        assert_eq!(mesh.n_devices(), 1);
        assert_eq!(mesh.axes(), &[]);
        assert_eq!(mesh.coord_of(0), Vec::<usize>::new());
        assert_eq!(mesh.device_of(&[]), 0);
        assert_eq!(mesh.group_along(DimKind::Tp, &[]), vec![0]);
        assert_eq!(mesh.group_along(DimKind::Ep, &[]), vec![0]);
        assert_eq!(mesh.stage_for_layer(0, 32), 0);
        assert_eq!(mesh.band_xfer_after(0, 32), None);
        assert_eq!(mesh.stage_devices(&[]), vec![0]);
    }

    #[test]
    fn single_and_empty_rect_have_same_shape_but_fresh_identity() {
        let single = DeviceMesh::single().unwrap();
        let empty_rect = DeviceMesh::rect(&[]).unwrap();
        assert_ne!(single, empty_rect);
        assert_eq!(single.axes(), empty_rect.axes());
        assert_eq!(single.n_devices(), empty_rect.n_devices());
        assert_eq!(single.coord_of(0), empty_rect.coord_of(0));
        assert_ne!(single.epoch(), empty_rect.epoch());
        assert_eq!(single.epoch(), single.clone().epoch());
    }

    #[test]
    fn pp_bands_and_boundary_hints_are_uniform() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 3)]).unwrap();
        assert_eq!(
            (0..6)
                .map(|l| mesh.stage_for_layer(l, 6))
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 1, 2, 2]
        );
        assert_eq!(
            mesh.band_xfer_after(1, 6),
            Some(CollectiveHint::BandXfer { src: 0, dst: 1 })
        );
        assert_eq!(
            mesh.band_xfer_after(3, 6),
            Some(CollectiveHint::BandXfer { src: 1, dst: 2 })
        );
        assert_eq!(mesh.band_xfer_after(5, 6), None);
        assert_eq!(mesh.stage_devices(&[1]), vec![1]);
    }

    #[test]
    fn composed_coordinates_groups_stages_and_squeeze() {
        let mesh =
            DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2), (DimKind::Ep, 2)]).unwrap();
        assert_eq!(mesh.n_devices(), 8);
        assert_eq!(mesh.coord_of(0), vec![0, 0, 0]);
        assert_eq!(mesh.coord_of(7), vec![1, 1, 1]);

        // The final (Ep) axis varies fastest in the row-major flattening.
        let coord = [1, 0, 1];
        assert_eq!(mesh.device_of(&coord), 5);
        assert_eq!(mesh.coord_of(5), coord);
        assert_eq!(mesh.group_along(DimKind::Tp, &coord), vec![5, 7]);
        assert_eq!(mesh.group_along(DimKind::Ep, &coord), vec![4, 5]);
        assert_eq!(mesh.stage_devices(&[1, 0, 0]), vec![4, 5, 6, 7]);

        for pp in 0..2 {
            for tp in 0..2 {
                for ep in 0..2 {
                    let coord = [pp, tp, ep];
                    assert_eq!(mesh.coord_of(mesh.device_of(&coord)), coord);
                }
            }
        }

        let degenerate =
            DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 1), (DimKind::Ep, 2)]).unwrap();
        assert_eq!(
            degenerate.squeezed().axes(),
            &[
                Axis {
                    kind: DimKind::Pp,
                    size: 2
                },
                Axis {
                    kind: DimKind::Ep,
                    size: 2
                },
            ]
        );
        assert_eq!(degenerate.epoch(), degenerate.squeezed().epoch());
    }

    #[test]
    fn coordinate_round_trip_holds_for_every_device() {
        let mesh =
            DeviceMesh::rect(&[(DimKind::Pp, 3), (DimKind::Tp, 2), (DimKind::Ep, 2)]).unwrap();
        for device in 0..mesh.n_devices() {
            assert_eq!(mesh.device_of(&mesh.coord_of(device)), device);
        }
    }

    #[test]
    fn rectangular_cardinality_overflow_is_rejected() {
        let error = DeviceMesh::rect(&[(DimKind::Pp, usize::MAX), (DimKind::Tp, 2)])
            .expect_err("rectangular cardinality must fail closed");
        assert_eq!(error, MeshError::CardinalityOverflow);
    }
}
