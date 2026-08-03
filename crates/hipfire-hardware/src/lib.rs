// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-GPU pipeline-parallel orchestration. Layer bands, boundary copy,
//! peer-access plumbing.
//!
//! # Threading invariant
//!
//! hipfire engine is **single-threaded for HIP work**. All `Gpu::*` methods
//! must be called from the same OS thread for the lifetime of the daemon
//! process. The `bind_thread()` helper assumes this.
//!
//! NOT supported in v1:
//! - Calling `Gpu::*` from rayon/tokio worker threads.
//! - HIP stream callbacks (`hipStreamAddCallback`) that touch `Gpu`.
//!
//! Future features adding background workers MUST:
//! 1. Add `gpu.bind_thread()?;` as the FIRST statement on entry.
//! 2. Run debug builds to catch silent mis-binds via the bind_thread invariant.
//! 3. Pass the multi-GPU coherence gate.

pub mod mesh;
pub use mesh::{Axis, CollectiveHint, DeviceMesh, DimKind, MeshEpoch};

use hip_bridge::{
    DeviceBuffer, Event, HipError, HipResult, HipRuntime, RcclComms,
    HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
use rdna_compute::{AllocationDomainId, DType, Gpu, GpuTensor};

/// Device-resolution knobs the hardware layer needs at `Gpus` construction
/// time. Extracted from `hipfire_runtime::config` so this crate is a leaf
/// (no dependency on the runtime). `from_env()` reads the same `HIPFIRE_*`
/// vars the runtime config does, byte-for-byte — so single-axis behavior is
/// unchanged. (The explicit-param seam for per-mesh override lands with
/// `resolve_mesh`.)
#[derive(Clone, Debug, Default)]
pub struct DeviceResolveOpts {
    pub tp_use_rccl: Option<bool>,
    pub devices: Option<String>,
    pub emulate_gpus: Option<usize>,
    pub allow_mixed_arch: bool,
    pub uniform_vram_tolerance_gb: Option<f32>,
}

impl DeviceResolveOpts {
    pub fn from_env() -> Self {
        Self {
            tp_use_rccl: std::env::var("HIPFIRE_TP_USE_RCCL")
                .ok()
                .as_deref()
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false")),
            devices: std::env::var("HIPFIRE_DEVICES").ok(),
            emulate_gpus: std::env::var("HIPFIRE_EMULATE_GPUS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 2),
            allow_mixed_arch: std::env::var("HIPFIRE_ALLOW_MIXED_ARCH").ok().as_deref()
                == Some("1"),
            uniform_vram_tolerance_gb: std::env::var("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB")
                .ok()
                .and_then(|s| s.parse().ok()),
        }
    }
}

/// Opaque epoch for a weight-allocation domain — one per logical rank.
///
/// Two allocation origins compare equal only when they share the same pool
/// epoch (i.e., they reference the same allocation-domain generation on the
/// same logical rank of the same mesh).
///
/// `WeightPoolEpoch` cannot be fabricated — it is produced only by
/// [`Gpus::weight_origin`], [`Gpus::weight_origin_in`], or
/// [`Gpus::single_weight_origin`], each of which extracts the opaque
/// [`AllocationDomainId`] from a live [`Gpu`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct WeightPoolEpoch(AllocationDomainId);

/// Private helper: create a [`WeightPoolEpoch`] from the live
/// [`AllocationDomainId`] on a [`Gpu`].  This is the only way to
/// construct a `WeightPoolEpoch` — external callers receive one from
/// the `Gpus` methods listed above.
fn epoch_from_domain(id: AllocationDomainId) -> WeightPoolEpoch {
    WeightPoolEpoch(id)
}

/// Identifies a weight-allocation event: which mesh instance, which logical
/// rank, which physical device, and which pool epoch within that rank.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WeightAllocationOrigin {
    mesh_epoch: MeshEpoch,
    logical_rank: usize,
    physical_device: i32,
    pool_epoch: WeightPoolEpoch,
}

impl WeightAllocationOrigin {
    /// The mesh epoch of the [`DeviceMesh`] that was bound at construction.
    pub fn mesh_epoch(&self) -> MeshEpoch {
        self.mesh_epoch
    }
    /// The logical rank within the mesh (0-based).
    pub fn logical_rank(&self) -> usize {
        self.logical_rank
    }
    /// The physical HIP device id that the logical rank maps to.
    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    /// The per-rank pool epoch that distinguishes this allocation domain
    /// generation from prior or subsequent ones on the same rank.
    pub fn pool_epoch(&self) -> WeightPoolEpoch {
        self.pool_epoch
    }
}

/// Errors from [`Gpus::weight_origin`] / [`Gpus::weight_origin_in`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightOriginError {
    /// The `Gpus` was not constructed from a [`DeviceMesh`]; call
    /// [`Gpus::from_mesh`] first.
    UnboundMesh,
    /// The given logical rank is out of range for this `Gpus`.
    UnknownRank(usize),
    /// The mesh passed to [`Gpus::weight_origin_in`] has a different epoch
    /// than the mesh bound to this `Gpus` at construction.
    MeshEpochMismatch,
}

impl std::fmt::Display for WeightOriginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightOriginError::UnboundMesh => {
                write!(f, "Gpus was not constructed from a DeviceMesh")
            }
            WeightOriginError::UnknownRank(r) => {
                write!(f, "logical rank {r} is out of range")
            }
            WeightOriginError::MeshEpochMismatch => {
                write!(
                    f,
                    "mesh epoch mismatch: provided mesh is not the same instance bound to this Gpus"
                )
            }
        }
    }
}

impl std::error::Error for WeightOriginError {}

/// Stream-event handoff returned by `Gpus::boundary_copy`. When the src
/// device has an active stream, `completion` holds a HIP event recorded
/// after the async peer copy; `Gpus::wait_boundary` makes the dst stream
/// wait on it. When the src device has no active stream, the sync
/// `memcpy_peer` already serializes the copy on the host and `completion`
/// is `None` — `wait_boundary` returns immediately in that case.
///
/// The `Option` is consumed (set to `None`) by `wait_boundary`; if a
/// `BoundaryEvent` with `completion: Some` is dropped without going through
/// `wait_boundary`, the `Drop` impl logs a leak warning. The HIP event
/// handle leaks in that case — destroying it requires a runtime reference
/// we don't store here.
pub struct BoundaryEvent {
    pub dst_dev: usize,
    completion: Option<Event>,
}

impl Drop for BoundaryEvent {
    fn drop(&mut self) {
        if self.completion.is_some() {
            eprintln!(
                "WARN: BoundaryEvent for dst_dev={} dropped without wait_boundary — \
                 HIP event handle leaked. Always pair boundary_copy with wait_boundary.",
                self.dst_dev,
            );
        }
    }
}

pub struct Gpus {
    /// RCCL communicators (one per rank), lazily initialized on the first
    /// `all_reduce_sum_*` call. Declared BEFORE `devices` so `Drop` tears
    /// down comms (via `ncclCommDestroy`) before the underlying HIP
    /// devices, which RCCL relies on. `None` means RCCL hasn't been used
    /// or `HIPFIRE_TP_USE_RCCL=0` forced the opt-out.
    rccl_comms: Option<RcclComms>,
    pub devices: Vec<Gpu>,
    /// Per-layer device id, length = n_layers.
    pub layer_to_device: Vec<u8>,
    /// Index of the first layer of each band, length = n_devices.
    pub band_starts: Vec<usize>,
    pub peer_access_enabled: bool,
    /// Variant 2 (Megatron/DeepSpeed/vLLM convention): `output_norm + lm_head`
    /// live on `dev_last`, not on dev_0. Removes the final `s.x` cross-device
    /// copy after the layer loop.
    pub output_device: usize,
    /// Per-device replicas of asym{2,3,4} KV rotation tables. Empty until
    /// the KV cache constructor (Stage 5) populates them.
    pub givens_cos_per_dev: Vec<GpuTensor>,
    pub givens_sin_per_dev: Vec<GpuTensor>,
    /// Peer-direct all-reduce scratch: `peer_ar_tmp[r][slot]` is a buffer on
    /// device `r` holding one OTHER rank's partial during
    /// [`Gpus::all_reduce_sum_f32_peer`]. Lazily allocated / grown to the largest
    /// `count` seen. Leaked on teardown (raw `DeviceBuffer`, no Drop-free).
    peer_ar_tmp: Vec<Vec<DeviceBuffer>>,
    peer_ar_tmp_bytes: usize,

    // ── mesh / weight-allocation identity ─────────────────────────────
    /// `Some(mesh.epoch())` when constructed via [`Gpus::from_mesh`];
    /// `None` for single / uniform / TP constructors.
    mesh_epoch: Option<MeshEpoch>,
}

const DEFAULT_VRAM_TOLERANCE_GB: f64 = 2.0;

impl Gpus {
    /// Construct `n_devices` `Gpu` instances bound to logical IDs taken from
    /// `HIPFIRE_DEVICES` (or the first N visible if unset). Layers are split
    /// uniformly: max-min ≤ 1 layer per band. Pre-flight VRAM check enforces
    /// arch match and bounded VRAM delta (override
    /// `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB`, default 2 GiB).
    pub fn init_uniform(n_devices: usize, n_layers: usize) -> HipResult<Self> {
        if n_devices == 0 {
            return Err(HipError::new(0, "init_uniform: n_devices must be >= 1"));
        }
        if n_layers < n_devices {
            return Err(HipError::new(
                0,
                &format!(
                    "init_uniform: n_layers ({n_layers}) < n_devices ({n_devices}) — \
                     each device must own at least one layer",
                ),
            ));
        }
        let device_ids = resolve_device_ids(n_devices)?;
        let devices = construct_devices(&device_ids)?;
        preflight_vram_with_opts(&devices, /*check_vram_delta=*/ true)?;
        let per_device = uniform_split_counts(n_devices, n_layers);
        Self::from_parts(devices, per_device, n_layers)
    }

    /// Explicit escape hatch for asymmetric VRAM / hand-tuned splits.
    /// Keeps arch-mismatch and per-device bind/free pre-flight checks, but
    /// skips the uniform VRAM-delta gate. `per_device` length determines
    /// `n_devices`; sum determines `n_layers`.
    pub fn init_layers(per_device: &[usize]) -> HipResult<Self> {
        let n_devices = per_device.len();
        if n_devices == 0 {
            return Err(HipError::new(
                0,
                "init_layers: per_device must be non-empty",
            ));
        }
        if per_device.contains(&0) {
            return Err(HipError::new(
                0,
                "init_layers: each device must own ≥1 layer",
            ));
        }
        let n_layers: usize = per_device.iter().sum();
        let device_ids = resolve_device_ids(n_devices)?;
        let devices = construct_devices(&device_ids)?;
        // init_layers is the documented escape hatch for asymmetric VRAM
        // splits — the caller has declared the per-device counts, so skip
        // the VRAM-delta check (which would otherwise reject 32 GB MI50 +
        // 12 GB 6700 XT pairs out of the box). Arch-mismatch + per-device
        // bind+free probe still run.
        preflight_vram_with_opts(&devices, /*check_vram_delta=*/ false)?;
        Self::from_parts(devices, per_device.to_vec(), n_layers)
    }

    /// Reserved for v1.1 — automatic VRAM-weighted band assignment. For v1
    /// use `init_layers(...)` with hand-computed counts.
    pub fn init_vram_weighted(_n_devices: usize, _n_layers: usize) -> HipResult<Self> {
        Err(HipError::new(
            0,
            "init_vram_weighted: scheduled for v1.1; use init_layers(per_device) instead",
        ))
    }

    /// PP=1 back-compat path: wrap an existing single `Gpu` into a `Gpus`
    /// with all layers on dev 0. `output_device = 0`.
    ///
    /// The returned instance is **unbound** ([`weight_origin`] returns
    /// [`WeightOriginError::UnboundMesh`]).
    pub fn single(gpu: Gpu, n_layers: usize) -> Self {
        Self {
            rccl_comms: None,
            devices: vec![gpu],
            layer_to_device: vec![0; n_layers],
            band_starts: vec![0],
            peer_access_enabled: false,
            output_device: 0,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
            peer_ar_tmp: Vec::new(),
            peer_ar_tmp_bytes: 0,
            mesh_epoch: None,
        }
    }

    /// Tensor-parallel constructor: bring up `tp_size` devices that each run
    /// **every** layer (PP=1), sharded within-layer per a `ShardConfig`.
    ///
    /// Distinct from `init_uniform` (which bands layers across devices for
    /// pipeline parallelism): here `layer_to_device = [0; n_layers]` and
    /// `band_starts = [0, n_layers, …]` (device 0 "owns" all layers in the
    /// PP sense; bands ≥1 are empty) so PP-oriented helpers stay well-defined,
    /// while the TP forward path ignores the layer-band map and dispatches
    /// every layer on every rank. `output_device = 0` — the replicated
    /// lm_head lives on every rank and sampling reads rank 0 by convention
    /// (TP plan §3.5 / Stage 7).
    ///
    /// The Q/KV-head divisibility check lives on `ShardConfig::validate`
    /// (called at model load once head counts are known); this constructor
    /// only validates the device count. Pre-flight runs the arch-match +
    /// VRAM-delta gate (TP ranks are identical cards, so the uniform delta
    /// check applies).
    pub fn init_tp(tp_size: usize, n_layers: usize) -> HipResult<Self> {
        if tp_size == 0 {
            return Err(HipError::new(0, "init_tp: tp_size must be >= 1"));
        }
        if n_layers == 0 {
            return Err(HipError::new(0, "init_tp: n_layers must be >= 1"));
        }
        let device_ids = resolve_device_ids(tp_size)?;
        let devices = construct_devices(&device_ids)?;
        preflight_vram_with_opts(&devices, /*check_vram_delta=*/ true)?;

        // PP=1 TP topology: every device runs every layer. Encode the layer
        // map so PP helpers see device 0 owning all layers and devices ≥1
        // owning empty bands.
        let mut band_starts = vec![0usize; tp_size];
        for b in band_starts.iter_mut().skip(1) {
            *b = n_layers;
        }
        Ok(Self {
            rccl_comms: None,
            devices,
            layer_to_device: vec![0u8; n_layers],
            band_starts,
            peer_access_enabled: false,
            output_device: 0,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
            peer_ar_tmp: Vec::new(),
            peer_ar_tmp_bytes: 0,
            mesh_epoch: None,
        })
    }

    /// Build a `Gpus` from a resolved single-axis [`DeviceMesh`], delegating to
    /// the existing per-axis primitive so the resulting layer layout is
    /// byte-identical to the pre-mesh loader paths:
    /// - `Ep` axis → [`init_tp`] (every device runs every layer, MoE experts sharded)
    /// - `Tp` axis → [`init_uniform`] (matches today's `TpModel::load`; TP bands are
    ///   vestigial but preserved)
    /// - `Pp` axis → [`init_uniform`] (uniform layer bands across stages)
    ///
    /// Only invoked for multi-device meshes on the loader path. A single-device
    /// (1×1) mesh never reaches here — the daemon's single-GPU branch keeps its
    /// bare `Gpu` + `load_model`. Precedence `Ep > Tp > Pp` matches the daemon
    /// routing chain; the daemon guarantees at most one axis is >1, so the order
    /// only fixes a convention.
    ///
    /// Binds [`mesh.epoch()`](DeviceMesh::epoch) and issues a distinct
    /// [`WeightPoolEpoch`] for each logical rank.
    pub fn from_mesh(mesh: &DeviceMesh, n_layers: usize) -> HipResult<Self> {
        let mut gpus = if mesh.has_axis(DimKind::Ep) {
            Self::init_tp(mesh.size_of(DimKind::Ep), n_layers)?
        } else if mesh.has_axis(DimKind::Tp) {
            Self::init_uniform(mesh.size_of(DimKind::Tp), n_layers)?
        } else if mesh.has_axis(DimKind::Pp) {
            Self::init_uniform(mesh.size_of(DimKind::Pp), n_layers)?
        } else {
            return Err(HipError::new(
                0,
                "from_mesh: single-device (1×1) mesh has no multi-device axis; \
                 use Gpus::single / load_model for the single-GPU path",
            ));
        };
        Self::validate_mesh_device_count(mesh, gpus.devices.len())?;
        gpus.mesh_epoch = Some(mesh.epoch());
        Ok(gpus)
    }

    /// Bidirectional `hipDeviceEnablePeerAccess` between every pair of
    /// devices. Returns `Ok(true)` if every leg succeeded; `Ok(false)` if
    /// any pair reports `hipDeviceCanAccessPeer = 0` or
    /// `hipErrorPeerAccessUnsupported = 217` — orchestrator falls back to
    /// host-staged copies in that case. PP=1 short-circuits to `Ok(true)`.
    ///
    /// **MUST be called AFTER all peer-accessible allocations are live.**
    /// On ROCm 6.4.3 / gfx1100 we observed that `hipDeviceEnablePeerAccess`
    /// does not retroactively map allocations made after the enable call:
    /// `hipMemcpyPeer` then silently returns `hipSuccess` while writing
    /// nothing to dst. The supported flow is: `init_uniform` → load weights
    /// → KV-cache alloc → `enable_peer_all` → forward. Without
    /// `enable_peer_all`, peer copies still work via HIP's transparent
    /// host-staging — slower, but correct.
    ///
    /// Partial-success state is sticky: hipDeviceDisablePeerAccess is not
    /// wrapped, so pairs we already enabled stay enabled. We deliberately
    /// keep iterating past a failed pair so that *capable* pairs in an
    /// N≥3 topology still get peer-copy even when one edge is unsupported.
    /// `Ok(false)` means "at least one pair could not be enabled"; the
    /// global `peer_access_enabled` flag mirrors that. Functional impact
    /// of a `false` return is small — `boundary_copy` falls through to
    /// HIP's transparent host-staging on un-enabled pairs either way.
    pub fn enable_peer_all(&mut self) -> HipResult<bool> {
        let n = self.devices.len();
        if n <= 1 {
            self.peer_access_enabled = true;
            return Ok(true);
        }
        let mut all_ok = true;
        for i in 0..n {
            self.devices[i].bind_thread()?;
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Emulated dual-GPU (HIPFIRE_EMULATE_GPUS): two logical ranks
                // aliased onto the same physical device_id. A peer query/enable
                // for device == peer errors on some ROCm versions, which would
                // abort the load; skip it and leave peer access disabled so
                // boundary_copy uses the valid same-device d2d fallback
                // (memcpy_peer with src == dst). Inert on real multi-GPU, where
                // distinct devices never share a device_id.
                if self.devices[i].device_id == self.devices[j].device_id {
                    all_ok = false;
                    continue;
                }
                if !self.devices[i]
                    .hip
                    .can_access_peer(self.devices[i].device_id, self.devices[j].device_id)?
                {
                    all_ok = false;
                    continue;
                }
                match self.devices[i]
                    .hip
                    .enable_peer_access(self.devices[j].device_id)
                {
                    Ok(()) => {}
                    // ffi.rs already converts 704 → Ok(()); this arm is
                    // belt-and-suspenders against ROCm versions where the
                    // driver returns 704 through a different code path.
                    Err(e) if e.code == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
                    Err(e) if e.code == HIP_ERROR_PEER_ACCESS_UNSUPPORTED => {
                        all_ok = false;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        self.peer_access_enabled = all_ok;
        Ok(all_ok)
    }

    #[inline]
    pub fn device_for_layer(&self, layer_idx: usize) -> usize {
        self.layer_to_device[layer_idx] as usize
    }

    /// True when the layer at `layer_idx + 1` lives on a different device
    /// than `layer_idx`. False at the last layer (no successor).
    #[inline]
    pub fn is_band_boundary(&self, layer_idx: usize) -> bool {
        let next = layer_idx + 1;
        next < self.layer_to_device.len()
            && self.layer_to_device[next] != self.layer_to_device[layer_idx]
    }

    #[inline]
    pub fn output_device(&self) -> usize {
        self.output_device
    }

    /// Async cross-device copy. Enqueues `hipMemcpyPeerAsync` on the src
    /// device's active stream (or null if unset) and records a completion
    /// event the caller awaits via `wait_boundary` before issuing the next
    /// dispatch on `dst_dev`. HIP transparently host-stages when peer
    /// access is unavailable; correctness holds either way.
    pub fn boundary_copy(
        &self,
        src_dev: usize,
        dst_dev: usize,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
        n_bytes: usize,
    ) -> HipResult<BoundaryEvent> {
        if src_dev == dst_dev {
            return Err(HipError::new(
                0,
                "boundary_copy: src_dev == dst_dev (use memcpy_dtod instead)",
            ));
        }
        if src_dev >= self.devices.len() || dst_dev >= self.devices.len() {
            return Err(HipError::new(
                0,
                &format!(
                    "boundary_copy: src_dev={src_dev} or dst_dev={dst_dev} out of \
                     range (n_devices={})",
                    self.devices.len(),
                ),
            ));
        }
        let src_gpu = &self.devices[src_dev];
        src_gpu.bind_thread()?;
        let src_dev_id = src_gpu.device_id;
        let dst_dev_id = self.devices[dst_dev].device_id;
        match src_gpu.active_stream.as_ref() {
            Some(stream) => {
                src_gpu
                    .hip
                    .memcpy_peer_async(dst, dst_dev_id, src, src_dev_id, n_bytes, stream)?;
                let event = src_gpu.hip.event_create()?;
                match src_gpu.hip.event_record(&event, Some(stream)) {
                    Ok(()) => Ok(BoundaryEvent {
                        dst_dev,
                        completion: Some(event),
                    }),
                    Err(e) => {
                        let _ = src_gpu.hip.event_destroy(event);
                        Err(e)
                    }
                }
            }
            None => {
                // Sync path: memcpy_peer blocks on host until the copy
                // lands. No event needed — recording into the HIP null
                // stream is fragile across ROCm versions; skip it and
                // signal "already done" via completion: None.
                src_gpu
                    .hip
                    .memcpy_peer(dst, dst_dev_id, src, src_dev_id, n_bytes)?;
                Ok(BoundaryEvent {
                    dst_dev,
                    completion: None,
                })
            }
        }
    }

    /// Stream-event handoff: makes dst's active stream (or null) wait on
    /// the completion event recorded by `boundary_copy`. Consumes the
    /// `BoundaryEvent` and destroys the underlying HIP event regardless
    /// of the wait result. If `completion` is `None` (sync copy already
    /// serialized on host), returns immediately without touching HIP.
    pub fn wait_boundary(&self, mut evt: BoundaryEvent) -> HipResult<()> {
        if evt.dst_dev >= self.devices.len() {
            return Err(HipError::new(
                0,
                &format!(
                    "wait_boundary: dst_dev={} out of range (n_devices={})",
                    evt.dst_dev,
                    self.devices.len(),
                ),
            ));
        }
        let Some(event) = evt.completion.take() else {
            return Ok(());
        };
        let dst_gpu = &self.devices[evt.dst_dev];
        dst_gpu.bind_thread()?;
        let wait_result = if let Some(stream) = dst_gpu.active_stream.as_ref() {
            dst_gpu.hip.stream_wait_event(stream, &event)
        } else {
            // No dst stream: host-block on the event so the next null-stream
            // dispatch on dst is ordered after the peer copy.
            dst_gpu.hip.event_synchronize(&event)
        };
        let destroy_result = dst_gpu.hip.event_destroy(event);
        wait_result.and(destroy_result)
    }

    fn from_parts(devices: Vec<Gpu>, per_device: Vec<usize>, n_layers: usize) -> HipResult<Self> {
        debug_assert_eq!(per_device.iter().sum::<usize>(), n_layers);
        debug_assert_eq!(per_device.len(), devices.len());
        let n_devices = devices.len();
        let mut layer_to_device = Vec::with_capacity(n_layers);
        let mut band_starts = Vec::with_capacity(n_devices);
        let mut cursor = 0;
        for (dev_idx, &count) in per_device.iter().enumerate() {
            band_starts.push(cursor);
            for _ in 0..count {
                layer_to_device.push(dev_idx as u8);
            }
            cursor += count;
        }
        Ok(Self {
            rccl_comms: None,
            devices,
            layer_to_device,
            band_starts,
            peer_access_enabled: false,
            output_device: n_devices - 1,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
            peer_ar_tmp: Vec::new(),
            peer_ar_tmp_bytes: 0,
            mesh_epoch: None,
        })
    }

    // ──────────────────────────────────────────────────────────────────
    // Tensor-parallel collectives (RCCL-backed). See
    // docs/plans/multi-gpu-tp-a3b.md §3.3 and the comm baseline at
    // docs/investigations/2026-05-28-tp-comm-baseline-hiptrx.md.
    // ──────────────────────────────────────────────────────────────────

    /// Lazily initialize RCCL communicators across all devices owned by
    /// this `Gpus`. Cached for process lifetime; subsequent calls are
    /// no-ops. `HIPFIRE_TP_USE_RCCL=0` short-circuits with a clear
    /// error so callers can fall through to a host-driven path (not
    /// yet implemented — Stage 2 follow-up).
    pub fn ensure_rccl(&mut self) -> HipResult<()> {
        if self.rccl_comms.is_some() {
            return Ok(());
        }
        if matches!(DeviceResolveOpts::from_env().tp_use_rccl, Some(false)) {
            return Err(HipError::new(
                0,
                "ensure_rccl: HIPFIRE_TP_USE_RCCL=0 — RCCL path opted out. \
                 Host-driven all-reduce fallback is not yet implemented \
                 (Stage 2 follow-up; see docs/plans/multi-gpu-tp-a3b.md).",
            ));
        }
        let device_ids: Vec<i32> = self.devices.iter().map(|d| d.device_id).collect();
        let comms = RcclComms::init_all(&device_ids).map_err(|e| {
            HipError::new(
                0,
                &format!(
                    "RcclComms::init_all(devices={:?}) failed: {}. \
                     Is librccl.so installed? On Debian/Ubuntu: \
                     `apt install rccl`; on ROCm install: \
                     `/opt/rocm/lib/librccl.so.1` must be present.",
                    device_ids, e
                ),
            )
        })?;
        self.rccl_comms = Some(comms);
        Ok(())
    }

    /// All-reduce-sum of f32 buffers across all ranks. `buffers[r]` must
    /// be a device pointer on `devices[r]` holding `count` f32 elements;
    /// after this call, each buffer holds the element-wise sum across
    /// all ranks. In-place (send == recv) — saves a memcpy and matches
    /// how the TP forward path uses the result.
    ///
    /// Requires each device to have an `active_stream` set (the stream
    /// the collective runs on). Synchronization is the caller's
    /// responsibility: this call enqueues the collective and returns
    /// immediately; the buffers are valid only after a subsequent
    /// `stream_synchronize` (or a downstream dispatch that's already
    /// ordered behind the same stream).
    ///
    /// `group` lists the global device ids participating (buffers are aligned
    /// to it: `buffers[k]` lives on `self.devices[group[k]]`). The RCCL path
    /// reduces over its single all-device communicator, so today it requires
    /// the **full** device set (`group == 0..n`); true sub-group reduction
    /// needs `ncclCommSplit` (Phase 5b). Pass `group = &(0..n)` for the 1D
    /// case — byte-identical to the previous all-devices behavior. For genuine
    /// sub-groups use `all_reduce_sum_f32_peer`, which is group-capable now.
    pub fn all_reduce_sum_f32(
        &mut self,
        group: &[usize],
        buffers: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        if buffers.len() != group.len() {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_f32: buffers.len()={} != group.len()={}",
                    buffers.len(),
                    group.len()
                ),
            ));
        }
        if group.len() != self.devices.len() {
            return Err(HipError::new(
                0,
                "all_reduce_sum_f32 (RCCL): sub-group reduction needs ncclCommSplit \
                 (Phase 5b); use all_reduce_sum_f32_peer for sub-groups.",
            ));
        }
        // Single-rank (TP=1) degenerate case: the all-reduce-sum over one
        // buffer is the identity — the buffer already holds the only rank's
        // partial. Short-circuit so the TP=1 EP path is a pure single-GPU
        // reference that exercises the full EP executor WITHOUT requiring
        // librccl (a 1-rank communicator would also work, but skipping it
        // keeps TP=1 dependency-free and the parity baseline trivially exact).
        if self.devices.len() == 1 {
            return Ok(());
        }
        self.ensure_rccl()?;

        // Borrow-check note: `self.rccl_comms.as_ref()` projects through
        // a single field, leaving `self.devices` independently
        // borrow-able for the per-rank stream lookup below.
        let rccl = self.rccl_comms.as_ref().expect("ensure_rccl populated it");

        rccl.group_start()
            .map_err(|e| HipError::new(0, &format!("ncclGroupStart: {e}")))?;
        for (r, buf) in buffers.iter().enumerate() {
            let dev = &self.devices[r];
            dev.bind_thread()?;
            let stream = dev.active_stream.as_ref().ok_or_else(|| {
                HipError::new(
                    0,
                    &format!(
                        "all_reduce_sum_f32: device {r} has no active_stream — \
                         set `gpus.devices[r].active_stream = Some(stream)` before calling.",
                    ),
                )
            })?;
            // SAFETY: `buf` is a live device buffer of `count` f32 on device
            // `r`, and `stream` is that device's active stream. (RCCL binding
            // became `unsafe fn` upstream.)
            unsafe {
                rccl.all_reduce_sum_f32(
                    r,
                    buf.as_ptr() as *const f32,
                    buf.as_ptr() as *mut f32,
                    count,
                    stream.raw_ptr(),
                )
            }
            .map_err(|e| HipError::new(0, &format!("ncclAllReduce rank={r}: {e}")))?;
        }
        rccl.group_end()
            .map_err(|e| HipError::new(0, &format!("ncclGroupEnd: {e}")))?;
        Ok(())
    }

    /// Ensure `peer_ar_tmp[r]` holds `n-1` buffers of at least `bytes` on each
    /// device. Lazily allocates; grows (freeing the old set) if `bytes` exceeds
    /// the current size. No-op for `n <= 1`.
    fn ensure_peer_ar_tmp(&mut self, bytes: usize) -> HipResult<()> {
        let n = self.devices.len();
        if n <= 1 {
            return Ok(());
        }
        if !self.peer_ar_tmp.is_empty() && self.peer_ar_tmp_bytes >= bytes {
            return Ok(());
        }
        // Free the old (too-small) set on its owning devices before regrowing.
        if !self.peer_ar_tmp.is_empty() {
            for (r, row) in std::mem::take(&mut self.peer_ar_tmp)
                .into_iter()
                .enumerate()
            {
                let _ = self.devices[r].bind_thread();
                for buf in row {
                    let _ = self.devices[r].hip.free(buf);
                }
            }
        }
        let mut all = Vec::with_capacity(n);
        for r in 0..n {
            self.devices[r].bind_thread()?;
            let mut row = Vec::with_capacity(n - 1);
            for _ in 0..(n - 1) {
                row.push(self.devices[r].hip.malloc(bytes)?);
            }
            all.push(row);
        }
        self.peer_ar_tmp = all;
        self.peer_ar_tmp_bytes = bytes;
        Ok(())
    }

    /// All-reduce-sum of f32 buffers across all ranks via **direct peer copy +
    /// local add** — bypassing RCCL. On consumer/prosumer RDNA P2P (no xGMI,
    /// e.g. hiptrx 4× gfx1201), `ncclAllReduce` costs ~40 ms/call for these
    /// small/medium messages regardless of NCCL_PROTO/CHANNELS/BUFFSIZE/
    /// SOCKET_IFNAME, while this path is ~1 ms. Used by EP prefill and TP; EP
    /// decode's tiny per-token reduce stays on RCCL (already fast). PP never
    /// all-reduces (it uses `boundary_copy` point-to-point).
    ///
    /// Algorithm (N-rank, race-free): **phase 1** copies every OTHER rank's
    /// ORIGINAL buffer into a local temp (all reads, no writes); a barrier
    /// (`wait_boundary`); **phase 2** adds the peer temps into the local buffer.
    /// All-reads-before-writes ⇒ no cross-device read/write race. `n==1` is the
    /// identity (no-op). Requires peer access (caller's `enable_peer_all`) for
    /// the fast P2P path; without it `boundary_copy` host-stages (slower but
    /// correct). In-place: `buffers[r]` is both input and output.
    ///
    /// `group` lists the global device ids participating; `buffers[k]` lives on
    /// `self.devices[group[k]]`. Unlike the RCCL path, this is **genuinely
    /// sub-group-capable** — it reduces only over `group` (peer copies + local
    /// add among those devices). Pass `group = &(0..n)` for the 1D all-devices
    /// case — byte-identical to the previous behavior.
    pub fn all_reduce_sum_f32_peer(
        &mut self,
        group: &[usize],
        buffers: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let g = group.len();
        if buffers.len() != g {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_f32_peer: buffers.len()={} != group.len()={g}",
                    buffers.len()
                ),
            ));
        }
        if g <= 1 {
            return Ok(());
        }
        let bytes = count * 4;
        // Sizes n-1 temp slots per physical device; a sub-group of size g uses
        // the first g-1 (g <= n ⇒ g-1 <= n-1), so no resize needed.
        self.ensure_peer_ar_tmp(bytes)?;

        // Phase 1: read every peer's ORIGINAL buffer into a local temp.
        let mut evts = Vec::with_capacity(g * (g - 1));
        for k in 0..g {
            let dev_k = group[k];
            let mut slot = 0usize;
            for m in 0..g {
                if m == k {
                    continue;
                }
                let dev_m = group[m];
                let evt = self.boundary_copy(
                    dev_m,
                    dev_k,
                    buffers[m],
                    &self.peer_ar_tmp[dev_k][slot],
                    bytes,
                )?;
                evts.push(evt);
                slot += 1;
            }
        }
        for evt in evts {
            self.wait_boundary(evt)?;
        }

        // Phase 2: add the peer temps into each rank's buffer.
        for k in 0..g {
            let dev_k = group[k];
            let dst = GpuTensor {
                buf: unsafe { buffers[k].alias() },
                shape: vec![count],
                dtype: DType::F32,
            };
            let srcs: Vec<GpuTensor> = (0..g - 1)
                .map(|slot| GpuTensor {
                    buf: unsafe { self.peer_ar_tmp[dev_k][slot].alias() },
                    shape: vec![count],
                    dtype: DType::F32,
                })
                .collect();
            self.devices[dev_k].bind_thread()?;
            for src in &srcs {
                self.devices[dev_k].add_inplace_f32(&dst, src)?;
            }
        }
        Ok(())
    }

    /// Int64 peer-direct all-reduce: sums `count` int64 elements in-place across
    /// `group`. Mirrors [`Self::all_reduce_sum_f32_peer`] exactly — same peer-copy
    /// structure, same scratch management — but operates on 8-byte int64 elements
    /// using `add_inplace_i64` for the local accumulation step.
    ///
    /// Used by the TP down collective in the reproducible MoE down scheme: each
    /// rank writes a S-scaled int64 partial, the partials are summed here (exact,
    /// no FP rounding), then `moe_i64_residual_to_f32` converts after.
    pub fn all_reduce_sum_i64_peer(
        &mut self,
        group: &[usize],
        buffers: &[&hip_bridge::DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let g = group.len();
        if buffers.len() != g {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_i64_peer: buffers.len()={} != group.len()={g}",
                    buffers.len()
                ),
            ));
        }
        if g <= 1 {
            return Ok(());
        }
        let bytes = count * 8; // 8 bytes per i64
        self.ensure_peer_ar_tmp(bytes)?;

        // Phase 1: copy every peer's ORIGINAL buffer into a local temp slot.
        let mut evts = Vec::with_capacity(g * (g - 1));
        for k in 0..g {
            let dev_k = group[k];
            let mut slot = 0usize;
            for m in 0..g {
                if m == k {
                    continue;
                }
                let dev_m = group[m];
                let evt = self.boundary_copy(
                    dev_m,
                    dev_k,
                    buffers[m],
                    &self.peer_ar_tmp[dev_k][slot],
                    bytes,
                )?;
                evts.push(evt);
                slot += 1;
            }
        }
        for evt in evts {
            self.wait_boundary(evt)?;
        }

        // Phase 2: add the peer temps into each rank's buffer (int64, exact).
        for k in 0..g {
            let dev_k = group[k];
            let dst_ptr = buffers[k].as_ptr();
            let src_ptrs: Vec<*mut std::ffi::c_void> = (0..g - 1)
                .map(|slot| self.peer_ar_tmp[dev_k][slot].as_ptr())
                .collect();
            self.devices[dev_k].bind_thread()?;
            for &src_ptr in &src_ptrs {
                self.devices[dev_k].add_inplace_i64(dst_ptr, src_ptr, count)?;
            }
        }
        Ok(())
    }

    /// Ensure every device has an `active_stream` — the FIFO stream the EP
    /// collective enqueues its per-rank work on. Idempotent (no-op if already
    /// set). The EP forward requires this; call it once after the ranks are
    /// constructed and before the first EP forward. Single-GPU / PP paths must
    /// NOT call it (they leave `active_stream = None` so memset stays
    /// synchronous).
    pub fn ensure_rank_streams(&mut self) -> HipResult<()> {
        for dev in self.devices.iter_mut() {
            dev.bind_thread()?;
            if dev.active_stream.is_none() {
                dev.active_stream = Some(dev.hip.stream_create()?);
            }
        }
        Ok(())
    }

    // ── weight-allocation origin ──────────────────────────────────────

    /// Assemble a [`WeightAllocationOrigin`] from already-resolved
    /// components.  Pure helper — no `&self`, no GPU access required.
    fn assemble_origin(
        mesh_epoch: MeshEpoch,
        logical_rank: usize,
        physical_device: i32,
        pool_epoch: WeightPoolEpoch,
    ) -> WeightAllocationOrigin {
        WeightAllocationOrigin {
            mesh_epoch,
            logical_rank,
            physical_device,
            pool_epoch,
        }
    }

    /// Validate that `mesh.n_devices()` matches the number of constructed
    /// GPU devices (`devices_len`).  Must be called after the per-axis
    /// primitive has been dispatched but before binding mesh/pool identities.
    fn validate_mesh_device_count(mesh: &DeviceMesh, devices_len: usize) -> HipResult<()> {
        let mesh_n = mesh.n_devices();
        if mesh_n != devices_len {
            return Err(HipError::new(
                0,
                &format!(
                    "from_mesh: mesh has {mesh_n} device{} but Gpus has {devices_len} device{}",
                    if mesh_n == 1 { "" } else { "s" },
                    if devices_len == 1 { "" } else { "s" },
                ),
            ));
        }
        Ok(())
    }

    /// Return the [`WeightAllocationOrigin`] for `rank`, or a descriptive
    /// error if this `Gpus` was never bound to a [`DeviceMesh`] or `rank` is
    /// out of range.
    ///
    /// The physical device id is always read from
    /// `self.devices[rank].device_id` — no synthetic fallback exists.
    pub fn weight_origin(&self, rank: usize) -> Result<WeightAllocationOrigin, WeightOriginError> {
        let mesh_epoch = self.mesh_epoch.ok_or(WeightOriginError::UnboundMesh)?;
        let dev = self
            .devices
            .get(rank)
            .ok_or(WeightOriginError::UnknownRank(rank))?;
        let physical_device = dev.device_id;
        let pool_epoch = epoch_from_domain(*dev.allocation_domain_id());
        Ok(Self::assemble_origin(
            mesh_epoch,
            rank,
            physical_device,
            pool_epoch,
        ))
    }

    // ── weight-origin helpers for single/mesh-aware queries ──────────

    /// Construct a [`WeightAllocationOrigin`] for a single GPU within a
    /// [`DeviceMesh`], without a `Gpus` wrapper.  The logical rank is always
    /// 0 (a single GPU is rank 0 of a 1-wide mesh).
    pub fn single_weight_origin(mesh: &DeviceMesh, gpu: &Gpu) -> WeightAllocationOrigin {
        Self::assemble_origin(
            mesh.epoch(),
            0,
            gpu.device_id,
            epoch_from_domain(*gpu.allocation_domain_id()),
        )
    }
    /// Return the [`WeightAllocationOrigin`] for `rank`, additionally
    /// validating that `mesh` matches the mesh bound to this `Gpus`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightOriginError::UnboundMesh`] when this `Gpus` was never
    /// bound to a [`DeviceMesh`].
    ///
    /// Returns [`WeightOriginError::UnknownRank`] when `rank` is out of range
    /// for `self.devices`.
    ///
    /// Returns [`WeightOriginError::MeshEpochMismatch`] when `mesh.epoch()`
    /// differs from the bound mesh's epoch.
    pub fn weight_origin_in(
        &self,
        mesh: &DeviceMesh,
        rank: usize,
    ) -> Result<WeightAllocationOrigin, WeightOriginError> {
        let mesh_epoch = self.mesh_epoch.ok_or(WeightOriginError::UnboundMesh)?;
        if mesh.epoch() != mesh_epoch {
            return Err(WeightOriginError::MeshEpochMismatch);
        }
        let dev = self
            .devices
            .get(rank)
            .ok_or(WeightOriginError::UnknownRank(rank))?;
        let physical_device = dev.device_id;
        let pool_epoch = epoch_from_domain(*dev.allocation_domain_id());
        Ok(Self::assemble_origin(
            mesh_epoch,
            rank,
            physical_device,
            pool_epoch,
        ))
    }
}

/// `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` selects the RCCL-free peer-direct
/// all-reduce for the EP MoE collective (no librccl dependency). Cached for
/// process lifetime.
pub fn ep_peer_allreduce_decode() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("HIPFIRE_EP_PEER_ALLREDUCE_DECODE").as_deref() == Ok("1"))
}

fn uniform_split_counts(n_devices: usize, n_layers: usize) -> Vec<usize> {
    let base = n_layers / n_devices;
    let rem = n_layers % n_devices;
    (0..n_devices)
        .map(|i| base + if i < rem { 1 } else { 0 })
        .collect()
}

/// Map each requested logical device id into the physical range
/// `[0, real_count)` by euclidean remainder. Used by `HIPFIRE_EMULATE_GPUS`
/// to alias N logical devices onto the (fewer) physical devices — e.g.
/// `[0, 1] -> [0, 0]` on a 1-GPU box, `[0, 1, 2, 3] -> [0, 1, 0, 1]` on a
/// 2-GPU box. A non-positive `real_count` is left untouched (no physical
/// devices to alias onto — the caller will surface the real error).
fn alias_ids(ids: &[i32], real_count: i32) -> Vec<i32> {
    if real_count <= 0 {
        return ids.to_vec();
    }
    ids.iter().map(|&id| id.rem_euclid(real_count)).collect()
}

/// Resolve the device IDs to use. Logical IDs post-`HIP_VISIBLE_DEVICES`:
/// `HIPFIRE_DEVICES=0,1` selects the first two HIP-visible devices. When
/// unset, takes the first `n_devices` visible IDs.
fn resolve_device_ids(n_devices: usize) -> HipResult<Vec<i32>> {
    let opts = DeviceResolveOpts::from_env();
    let ids: Vec<i32> = if let Some(ref s) = opts.devices {
        let parsed: Vec<i32> = s
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<i32>())
            .collect::<Result<_, _>>()
            .map_err(|e| HipError::new(0, &format!("HIPFIRE_DEVICES parse: {e}")))?;
        if parsed.len() < n_devices {
            return Err(HipError::new(
                0,
                &format!(
                    "HIPFIRE_DEVICES has {} ids but n_devices = {n_devices}",
                    parsed.len(),
                ),
            ));
        }
        parsed[..n_devices].to_vec()
    } else {
        (0..n_devices as i32).collect()
    };

    // Debug dual-GPU emulation: alias every logical id into the physical range
    // so a single card can serve an N-way PP/EP load. Applies to both the
    // explicit HIPFIRE_DEVICES list and the default 0..n so neither can leave
    // an out-of-range id. See config::emulate_gpus.
    if opts.emulate_gpus.is_some() {
        let real = HipRuntime::load()?.device_count()?;
        return Ok(alias_ids(&ids, real));
    }
    Ok(ids)
}

fn construct_devices(ids: &[i32]) -> HipResult<Vec<Gpu>> {
    let mut devices = Vec::with_capacity(ids.len());
    for &id in ids {
        devices.push(Gpu::init_with_device(id)?);
    }
    Ok(devices)
}

fn preflight_vram_with_opts(devices: &[Gpu], check_vram_delta: bool) -> HipResult<()> {
    if devices.is_empty() {
        return Ok(());
    }
    let arch0 = devices[0].arch.clone();
    let opts = DeviceResolveOpts::from_env();
    let allow_mixed = opts.allow_mixed_arch;
    let mut frees = Vec::with_capacity(devices.len());
    for d in devices {
        if d.arch != arch0 {
            if allow_mixed {
                eprintln!(
                    "preflight_vram: mixed-arch detected — dev 0 is {arch0}, dev {} is {}. \
                     Proceeding because HIPFIRE_ALLOW_MIXED_ARCH=1. \
                     Per-arch JIT cache will be populated on first run; boundary_copy uses \
                     hipMemcpyPeer / hipMemcpyPeerAsync which fall through to host-staging \
                     if peer access is unsupported by the pair (correctness holds either way).",
                    d.device_id, d.arch,
                );
            } else {
                return Err(HipError::new(
                    0,
                    &format!(
                        "preflight_vram: arch mismatch — dev 0 is {arch0}, dev {} is {}. \
                         Mixed-arch is not supported by default; set HIPFIRE_ALLOW_MIXED_ARCH=1 to override.",
                        d.device_id, d.arch,
                    ),
                ));
            }
        }
        d.bind_thread()?;
        let (free, _total) = d.hip.get_vram_info()?;
        frees.push(free);
    }
    if !check_vram_delta {
        return Ok(());
    }
    let max_free = *frees.iter().max().unwrap();
    let min_free = *frees.iter().min().unwrap();
    let delta_gb = (max_free - min_free) as f64 / 1e9;
    let tol_gb = opts
        .uniform_vram_tolerance_gb
        .map(|t| t as f64)
        .unwrap_or(DEFAULT_VRAM_TOLERANCE_GB);
    if delta_gb > tol_gb {
        return Err(HipError::new(
            0,
            &format!(
                "preflight_vram: VRAM delta {:.1} GiB exceeds tolerance {:.1} GiB. \
                 Override via HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB or use init_layers().",
                delta_gb, tol_gb,
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── assemble_origin / WeightAllocationOrigin ──────────────────────

    /// Private test-only fixture: construct a real single-GPU [`Gpus`]
    /// bound to a given [`DeviceMesh`].  Requires ROCm hardware; panics
    /// via `expect` when unavailable.  The caller controls the mesh so they
    /// can create distinct epochs for mismatch testing.
    fn make_bound_gpus(mesh: &DeviceMesh) -> Gpus {
        let gpu = Gpu::init_with_device(0)
            .expect("requires ROCm GPU — install ROCm or run with --skip ignored");
        let mut gpus = Gpus::single(gpu, 24);
        gpus.mesh_epoch = Some(mesh.epoch());
        gpus
    }

    /// assemble_origin composition — requires a live Gpu to produce an
    /// [`AllocationDomainId`] (no public fabricator exists).
    #[test]
    #[ignore = "requires ROCm GPU"]
    fn assemble_origin_constructs_correctly() {
        let gpu = Gpu::init_with_device(0).expect("ROCm GPU required");
        let mesh_epoch = DeviceMesh::single().epoch();
        let epoch = epoch_from_domain(*gpu.allocation_domain_id());
        let origin = Gpus::assemble_origin(mesh_epoch, 7, 42, epoch);
        assert_eq!(origin.mesh_epoch(), mesh_epoch);
        assert_eq!(origin.logical_rank(), 7);
        assert_eq!(origin.physical_device(), 42);
        assert_eq!(origin.pool_epoch(), epoch);
    }

    /// Equality observes all four fields — requires two distinct
    /// [`AllocationDomainId`] values from distinct Gpu instances.
    #[test]
    #[ignore = "requires ROCm GPU"]
    fn weight_allocation_origin_equality_observes_all_four_fields() {
        let gpu1 = Gpu::init_with_device(0).expect("ROCm GPU required");
        let gpu2 = Gpu::init_with_device(0).expect("ROCm GPU required");
        let me1 = DeviceMesh::single().epoch();
        let me2 = DeviceMesh::rect(&[(DimKind::Tp, 2)]).epoch();
        let epoch = epoch_from_domain(*gpu1.allocation_domain_id());
        let epoch_diff = epoch_from_domain(*gpu2.allocation_domain_id());
        let base = Gpus::assemble_origin(me1, 0, 0, epoch);
        assert_ne!(base, Gpus::assemble_origin(me2, 0, 0, epoch), "mesh epoch");
        assert_ne!(
            base,
            Gpus::assemble_origin(me1, 1, 0, epoch),
            "logical rank"
        );
        assert_ne!(
            base,
            Gpus::assemble_origin(me1, 0, 1, epoch),
            "physical device"
        );
        assert_ne!(
            base,
            Gpus::assemble_origin(me1, 0, 0, epoch_diff),
            "pool epoch"
        );
    }

    // ── validate_mesh_device_count (pure helper, no Gpu required) ──────

    #[test]
    fn validate_mesh_n_devices_accepts_single_axis() {
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 4)]);
        assert!(Gpus::validate_mesh_device_count(&mesh, 4).is_ok());
    }

    #[test]
    fn validate_mesh_n_devices_rejects_composed_2x2() {
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 2)]);
        let err = Gpus::validate_mesh_device_count(&mesh, 2).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("mesh has 4 devices but Gpus has 2 devices"),
            "error message: {msg}",
        );
    }

    // ── pre-existing pure CPU tests ────────────────────────────────────

    #[test]
    fn alias_ids_maps_into_physical_range() {
        assert_eq!(alias_ids(&[0, 1], 1), vec![0, 0]);
        assert_eq!(alias_ids(&[0, 1, 2, 3], 2), vec![0, 1, 0, 1]);
        assert_eq!(alias_ids(&[0, 0], 1), vec![0, 0]);
        assert_eq!(alias_ids(&[0, 1], 2), vec![0, 1]);
    }

    #[test]
    fn uniform_split_basic() {
        assert_eq!(uniform_split_counts(2, 24), vec![12, 12]);
        assert_eq!(uniform_split_counts(2, 25), vec![13, 12]);
        assert_eq!(uniform_split_counts(3, 64), vec![22, 21, 21]);
        assert_eq!(uniform_split_counts(4, 7), vec![2, 2, 2, 1]);
    }

    #[test]
    fn uniform_split_invariants() {
        for n_devices in 1..=6 {
            for n_layers in n_devices..=80 {
                let split = uniform_split_counts(n_devices, n_layers);
                assert_eq!(split.iter().sum::<usize>(), n_layers);
                let mn = *split.iter().min().unwrap();
                let mx = *split.iter().max().unwrap();
                assert!(mx - mn <= 1, "split {split:?} for {n_devices}/{n_layers}");
            }
        }
    }

    // ── WeightOriginError Display (pure, no Gpu required) ──────────────

    #[test]
    fn weight_origin_error_mesh_epoch_mismatch_display() {
        let err = WeightOriginError::MeshEpochMismatch;
        let msg = err.to_string();
        assert!(msg.contains("mesh epoch mismatch"), "error message: {msg}");
    }

    #[test]
    fn weight_origin_error_unbound_mesh_display() {
        let err = WeightOriginError::UnboundMesh;
        let msg = err.to_string();
        assert!(msg.contains("not constructed from"), "error message: {msg}");
    }

    #[test]
    fn weight_origin_error_unknown_rank_display() {
        let err = WeightOriginError::UnknownRank(42);
        let msg = err.to_string();
        assert!(
            msg.contains("rank") && msg.contains("42"),
            "error message: {msg}",
        );
    }

    // ── weight_origin / weight_origin_in / single_weight_origin ──────
    //
    // All tests below require a real Gpu initialized via
    // `Gpu::init_with_device(0)` → `HipRuntime::load()` → dlopen of
    // `libamdhip64.so`.  Without an AMD GPU + ROCm installed they are
    // skipped with `#[ignore = "requires ROCm GPU"]`.
    //
    // The `make_bound_gpus` fixture (above) wraps a bare `Gpu` into a
    // mesh-bound `Gpus` by mutating the private `mesh_epoch` field, so
    // the `weight_origin*` methods reach the rank check / epoch check
    // instead of short-circuiting at `UnboundMesh`.

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn gpu_allocation_domain_id_stable_through_drain_pool() {
        let mut gpu = Gpu::init_with_device(0).expect("ROCm GPU required");
        let before = *gpu.allocation_domain_id();
        gpu.drain_pool();
        let after = *gpu.allocation_domain_id();
        assert_eq!(
            before, after,
            "drain_pool must not change allocation_domain_id"
        );
    }

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn gpu_construction_assigns_unique_domain_ids() {
        let gpu0 = Gpu::init_with_device(0).expect("ROCm GPU required");
        let gpu1 = Gpu::init_with_device(0).expect("ROCm GPU required");
        assert_ne!(
            *gpu0.allocation_domain_id(),
            *gpu1.allocation_domain_id(),
            "two independently-constructed Gpu instances must have distinct allocation_domain_ids"
        );
    }

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn single_weight_origin_composes_origin_from_gpu_and_mesh() {
        let gpu = Gpu::init_with_device(0).expect("ROCm GPU required");
        let mesh = DeviceMesh::single();
        let origin = Gpus::single_weight_origin(&mesh, &gpu);
        assert_eq!(origin.mesh_epoch(), mesh.epoch());
        assert_eq!(origin.logical_rank(), 0);
        assert_eq!(origin.physical_device(), gpu.device_id);
        let expected_epoch = epoch_from_domain(*gpu.allocation_domain_id());
        assert_eq!(origin.pool_epoch(), expected_epoch);

        // Round-trip: extract the domain id from the origin via the
        // private epoch_from_domain helper — external callers cannot
        // fabricate a matching WeightPoolEpoch without it.
        let roundtrip = epoch_from_domain(*gpu.allocation_domain_id());
        assert_eq!(origin.pool_epoch(), roundtrip);
    }

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn weight_origin_unbound_mesh_error() {
        let gpu = Gpu::init_with_device(0).expect("ROCm GPU required");
        let gpus = Gpus::single(gpu, 24);
        let err = gpus.weight_origin(0).unwrap_err();
        assert_eq!(err, WeightOriginError::UnboundMesh);
    }

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn weight_origin_unknown_rank_error() {
        let mesh = DeviceMesh::single();
        let gpus = make_bound_gpus(&mesh);
        let err = gpus.weight_origin(99).unwrap_err();
        assert_eq!(err, WeightOriginError::UnknownRank(99));
    }

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn weight_origin_in_mesh_epoch_mismatch_error() {
        let mesh = DeviceMesh::single();
        let other_mesh = DeviceMesh::single();
        let gpus = make_bound_gpus(&mesh);
        let err = gpus.weight_origin_in(&other_mesh, 0).unwrap_err();
        assert_eq!(err, WeightOriginError::MeshEpochMismatch);
    }

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn weight_origin_in_unknown_rank_error() {
        let mesh = DeviceMesh::single();
        let gpus = make_bound_gpus(&mesh);
        let err = gpus.weight_origin_in(&mesh, 99).unwrap_err();
        assert_eq!(err, WeightOriginError::UnknownRank(99));
    }

    #[test]
    #[ignore = "requires ROCm GPU"]
    fn weight_origin_in_valid_bound_case() {
        let mesh = DeviceMesh::single();
        let gpus = make_bound_gpus(&mesh);
        let origin = gpus.weight_origin_in(&mesh, 0).expect("bound + valid rank");
        assert_eq!(origin.mesh_epoch(), mesh.epoch());
        assert_eq!(origin.logical_rank(), 0);
        assert_eq!(origin.physical_device(), gpus.devices[0].device_id);
        let expected_epoch = epoch_from_domain(*gpus.devices[0].allocation_domain_id());
        assert_eq!(origin.pool_epoch(), expected_epoch);
    }
}
