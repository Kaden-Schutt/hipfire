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

use hip_bridge::{
    DeviceBuffer, Event, HipError, HipResult, RcclComms, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    HIP_ERROR_PEER_ACCESS_UNSUPPORTED, HIP_EVENT_DISABLE_TIMING, HIP_EVENT_RELEASE_TO_SYSTEM,
};
use rdna_compute::{DType, Gpu, GpuTensor};

mod mesh;
pub use mesh::{Axis, CollectiveHint, DeviceMesh, DimKind, MeshEpoch, MeshError};

/// Device-resolution knobs supplied by the resolved process/runtime config
/// when constructing a [`Gpus`] owner.
///
/// `devices` contains the post-visibility logical HIP IDs produced by
/// `hipfire-config`, not the original physical ROCr selectors. Hardware
/// consumes this value as-is and never rereads process environment.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceResolveOpts {
    pub tp_use_rccl: Option<bool>,
    pub devices: Option<String>,
    pub allow_mixed_arch: bool,
    pub uniform_vram_tolerance_gb: Option<f32>,
}

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

/// Opaque unique non-clone peer-reduce scratch lease.
///
/// Exactly `N-1` buffers per rank are allocated under this lease; `N=4`
/// yields 3 per rank and 12 total. Acquisition is exclusive — a second
/// owner is rejected. The lease never clones/copies; ownership is
/// transferred by move. Validation of id/rank_count/bytes is performed on
/// every leased reduce/release.
pub struct PeerReduceScratchLease {
    id: u64,
    bytes: usize,
    rank_count: usize,
    _private: (),
}

impl std::fmt::Debug for PeerReduceScratchLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerReduceScratchLease")
            .field("id", &self.id)
            .field("bytes", &self.bytes)
            .field("rank_count", &self.rank_count)
            .finish()
    }
}

/// Pure helper: per-device peer scratch bytes for `rank_count` ranks and
/// `requested_bytes` per reduction. Returns `None` on overflow or
/// zero ranks. For `N=4`, per-rank is `3*requested_bytes` and total is
/// `12*requested_bytes`.
#[inline]
pub fn peer_reduce_scratch_bytes_per_rank(
    rank_count: usize,
    requested_bytes: usize,
) -> Option<usize> {
    if rank_count == 0 {
        return None;
    }
    let per_rank = rank_count.checked_sub(1)?.checked_mul(requested_bytes)?;
    Some(per_rank)
}

/// Pure helper: total peer scratch bytes across all ranks.
#[inline]
pub fn peer_reduce_scratch_total_bytes(rank_count: usize, requested_bytes: usize) -> Option<usize> {
    let per_rank = peer_reduce_scratch_bytes_per_rank(rank_count, requested_bytes)?;
    rank_count.checked_mul(per_rank)
}

/// Internal active lease record stored inside `Gpus` while a lease is live.
#[derive(Debug)]
struct ActivePeerLease {
    id: u64,
    bytes: usize,
    rank_count: usize,
}

pub struct Gpus {
    /// RCCL communicators (one per rank), lazily initialized on the first
    /// `all_reduce_sum_*` call. Declared BEFORE `devices` so `Drop` tears
    /// down comms (via `ncclCommDestroy`) before the underlying HIP
    /// devices, which RCCL relies on. `None` means RCCL hasn't been used.
    rccl_comms: Option<RcclComms>,
    /// Resolved at construction from the process/runtime config. Keeping this
    /// decision on the owner prevents a later environment mutation from
    /// changing the collective route.
    use_rccl: bool,
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
    /// `count` seen. Explicitly reclaimed via [`Gpus::free_peer_reduce_scratch`].
    peer_ar_tmp: Vec<Vec<DeviceBuffer>>,
    peer_ar_tmp_bytes: usize,
    /// Persistent pageable host staging for the fallback host all-reduce
    /// ([`Gpus::all_reduce_sum_f32_host`]). Outer vec is per-rank, inner vec
    /// is `count` f32 per rank. Grown to the largest `count` seen and reused
    /// without reallocation on steady state; drops normally.
    host_ar_tmp: Vec<Vec<f32>>,
    /// Unique peer-rooted scratch lease (TP4 EP). `None` when no lease is
    /// active. When `Some`, `peer_lease_buffers[r]` holds `N-1` buffers per
    /// rank at least `peer_lease.bytes` and leased reduces must use that
    /// buffer set without allocation/growth. A failed release quarantines
    /// the scratch so no future owner can use a partially freed set.
    active_peer_lease: Option<ActivePeerLease>,
    peer_lease_buffers: Vec<Vec<DeviceBuffer>>,
    peer_lease_next_id: u64,
    peer_lease_quarantined: bool,
    /// One process-lifetime dependency event per rank for peer-consumer
    /// collectives. Re-recording avoids 86 create/destroy pairs per DS4 token.
    rank_barrier_events: Vec<Event>,
    /// One 8-byte system-visible epoch per rank for the exact-gated gfx1201
    /// TP3/TP4 graph route. Each captured barrier advances the epoch.
    tp_graph_signals: Vec<DeviceBuffer>,
    tp_graph_barrier_count: usize,
    tp_graph_capture_epoch: usize,
}

const DEFAULT_VRAM_TOLERANCE_GB: f64 = 2.0;

impl Gpus {
    /// Construct `n_devices` `Gpu` instances bound to logical IDs derived from
    /// synchronized `hardware.devices` visibility (or the first N visible if
    /// unset). Layers are split
    /// uniformly: max-min ≤ 1 layer per band. Pre-flight VRAM check enforces
    /// arch match and bounded VRAM delta (override
    /// `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB`, default 2 GiB).
    pub fn init_uniform(
        opts: &DeviceResolveOpts,
        n_devices: usize,
        n_layers: usize,
    ) -> HipResult<Self> {
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
        let device_ids = resolve_device_ids(n_devices, opts)?;
        let devices = construct_devices(&device_ids)?;
        preflight_vram_with_opts(&devices, /*check_vram_delta=*/ true, opts)?;
        let per_device = uniform_split_counts(n_devices, n_layers);
        Self::from_parts(devices, per_device, n_layers, opts)
    }
    /// skips the uniform VRAM-delta gate. `per_device` length determines
    /// `n_devices`; sum determines `n_layers`.
    pub fn init_layers(opts: &DeviceResolveOpts, per_device: &[usize]) -> HipResult<Self> {
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
        let device_ids = resolve_device_ids(n_devices, opts)?;
        let devices = construct_devices(&device_ids)?;
        // init_layers is the documented escape hatch for asymmetric VRAM
        // splits — the caller has declared the per-device counts, so skip
        // the VRAM-delta check (which would otherwise reject 32 GB MI50 +
        // 12 GB 6700 XT pairs out of the box). Arch-mismatch + per-device
        // bind+free probe still run.
        preflight_vram_with_opts(&devices, /*check_vram_delta=*/ false, opts)?;
        Self::from_parts(devices, per_device.to_vec(), n_layers, opts)
    }

    pub fn init_vram_weighted(
        _opts: &DeviceResolveOpts,
        _n_devices: usize,
        _n_layers: usize,
    ) -> HipResult<Self> {
        Err(HipError::new(
            0,
            "init_vram_weighted: scheduled for v1.1; use init_layers(per_device) instead",
        ))
    }

    /// PP=1 back-compat path: wrap an existing single `Gpu` into a `Gpus`
    /// with all layers on dev 0. `output_device = 0`.
    pub fn single(opts: &DeviceResolveOpts, gpu: Gpu, n_layers: usize) -> Self {
        Self {
            rccl_comms: None,
            use_rccl: opts.tp_use_rccl.unwrap_or(true),
            devices: vec![gpu],
            layer_to_device: vec![0; n_layers],
            band_starts: vec![0],
            peer_access_enabled: false,
            output_device: 0,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
            peer_ar_tmp: Vec::new(),
            peer_ar_tmp_bytes: 0,
            host_ar_tmp: Vec::new(),
            active_peer_lease: None,
            peer_lease_buffers: Vec::new(),
            peer_lease_next_id: 0,
            peer_lease_quarantined: false,
            rank_barrier_events: Vec::new(),
            tp_graph_signals: Vec::new(),
            tp_graph_barrier_count: 0,
            tp_graph_capture_epoch: 0,
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
    pub fn init_tp(opts: &DeviceResolveOpts, tp_size: usize, n_layers: usize) -> HipResult<Self> {
        if tp_size == 0 {
            return Err(HipError::new(0, "init_tp: tp_size must be >= 1"));
        }
        if n_layers == 0 {
            return Err(HipError::new(0, "init_tp: n_layers must be >= 1"));
        }
        let device_ids = resolve_device_ids(tp_size, opts)?;
        let devices = construct_devices(&device_ids)?;
        preflight_vram_with_opts(&devices, /*check_vram_delta=*/ true, opts)?;
        let band_starts = tp_band_starts(tp_size, n_layers);

        // PP=1 TP topology: every device runs every layer. Encode the layer
        // map so PP helpers see device 0 owning all layers and devices ≥1
        // owning empty bands.
        Ok(Self {
            rccl_comms: None,
            use_rccl: opts.tp_use_rccl.unwrap_or(true),
            devices,
            layer_to_device: vec![0u8; n_layers],
            band_starts,
            peer_access_enabled: false,
            output_device: 0,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
            peer_ar_tmp: Vec::new(),
            peer_ar_tmp_bytes: 0,
            host_ar_tmp: Vec::new(),
            active_peer_lease: None,
            peer_lease_buffers: Vec::new(),
            peer_lease_next_id: 0,
            peer_lease_quarantined: false,
            rank_barrier_events: Vec::new(),
            tp_graph_signals: Vec::new(),
            tp_graph_barrier_count: 0,
            tp_graph_capture_epoch: 0,
        })
    }

    /// Query whether every directed device pair supports peer access.
    /// Does not enable peer access or mutate peer state. Use when partial
    /// activation is unsafe (ROCm does not map later allocations).
    pub fn can_access_peer_all(&self) -> HipResult<bool> {
        let n = self.devices.len();
        if n <= 1 {
            return Ok(true);
        }
        for i in 0..n {
            self.devices[i].bind_thread()?;
            for j in 0..n {
                if i == j {
                    continue;
                }
                if !self.devices[i]
                    .hip
                    .can_access_peer(self.devices[i].device_id, self.devices[j].device_id)?
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
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

    /// Enqueue an all-rank producer barrier using process-lifetime events.
    ///
    /// Each rank records its event after producing a peer-visible tensor, then
    /// every other rank waits before launching its fused peer consumer. Events
    /// are deliberately reused: HIP wait semantics bind to the most recent
    /// record visible when the wait is enqueued, and each next record follows
    /// the prior consumer on the same FIFO stream.
    pub fn barrier_rank_streams_reuse(&mut self) -> HipResult<()> {
        if self.tp_graph_signals.len() == self.devices.len()
            && matches!(self.devices.len(), 3 | 4)
            && self.devices.iter().all(|device| device.graphs.capture_mode)
        {
            return self.capture_tp_graph_barrier();
        }

        let n = self.devices.len();
        if self.rank_barrier_events.is_empty() {
            let mut events = Vec::with_capacity(n);
            for rank in 0..n {
                let gpu = &self.devices[rank];
                gpu.bind_thread()?;
                match gpu
                    .hip
                    .event_create_with_flags(HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM)
                {
                    Ok(event) => events.push(event),
                    Err(error) => {
                        for (owner, event) in events.drain(..).enumerate() {
                            let _ = self.devices[owner].hip.event_destroy(event);
                        }
                        return Err(error);
                    }
                }
            }
            self.rank_barrier_events = events;
        } else if self.rank_barrier_events.len() != n {
            return Err(HipError::new(
                0,
                &format!(
                    "barrier_rank_streams_reuse: event count {} != device count {n}",
                    self.rank_barrier_events.len()
                ),
            ));
        }

        for rank in 0..n {
            let gpu = &self.devices[rank];
            gpu.bind_thread()?;
            let stream = gpu.active_stream.as_ref().ok_or_else(|| {
                HipError::new(
                    0,
                    &format!("barrier_rank_streams_reuse: device {rank} has no active_stream"),
                )
            })?;
            gpu.hip
                .event_record(&self.rank_barrier_events[rank], Some(stream))?;
        }

        for destination in 0..n {
            let gpu = &self.devices[destination];
            gpu.bind_thread()?;
            let stream = gpu.active_stream.as_ref().expect("validated above");
            for source in 0..n {
                if source != destination {
                    gpu.hip
                        .stream_wait_event(stream, &self.rank_barrier_events[source])?;
                }
            }
        }
        Ok(())
    }

    /// One-way stream-event handoff from a rank that produced peer-visible
    /// state to every other rank that will consume it next.
    ///
    /// Unlike [`Self::barrier_rank_streams_reuse`], this does not wait for the
    /// destination ranks and therefore can be inserted between an owner-first
    /// producer launch and the remaining peer consumers without deadlocking.
    /// The event uses a system-scope release and is reused with FIFO stream
    /// ordering, matching the full-rank barrier's lifetime contract.
    pub fn handoff_rank_stream_reuse(&mut self, source: usize) -> HipResult<()> {
        let n = self.devices.len();
        if source >= n {
            return Err(HipError::new(
                0,
                &format!("handoff_rank_stream_reuse: source={source} out of range (n_devices={n})"),
            ));
        }
        if self.rank_barrier_events.is_empty() {
            let mut events = Vec::with_capacity(n);
            for rank in 0..n {
                let gpu = &self.devices[rank];
                gpu.bind_thread()?;
                match gpu
                    .hip
                    .event_create_with_flags(HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM)
                {
                    Ok(event) => events.push(event),
                    Err(error) => {
                        for (owner, event) in events.drain(..).enumerate() {
                            let _ = self.devices[owner].hip.event_destroy(event);
                        }
                        return Err(error);
                    }
                }
            }
            self.rank_barrier_events = events;
        } else if self.rank_barrier_events.len() != n {
            return Err(HipError::new(
                0,
                &format!(
                    "handoff_rank_stream_reuse: event count {} != device count {n}",
                    self.rank_barrier_events.len()
                ),
            ));
        }

        let producer = &self.devices[source];
        producer.bind_thread()?;
        let producer_stream = producer.active_stream.as_ref().ok_or_else(|| {
            HipError::new(
                0,
                &format!("handoff_rank_stream_reuse: source device {source} has no active_stream"),
            )
        })?;
        producer
            .hip
            .event_record(&self.rank_barrier_events[source], Some(producer_stream))?;

        for destination in 0..n {
            if destination == source {
                continue;
            }
            let consumer = &self.devices[destination];
            consumer.bind_thread()?;
            let consumer_stream = consumer.active_stream.as_ref().ok_or_else(|| {
                HipError::new(
                    0,
                    &format!(
                        "handoff_rank_stream_reuse: destination device {destination} has no active_stream"
                    ),
                )
            })?;
            consumer
                .hip
                .stream_wait_event(consumer_stream, &self.rank_barrier_events[source])?;
        }
        Ok(())
    }

    /// Allocate one fixed gfx1201 TP3/TP4 graph epoch before peer access is
    /// enabled. ROCm signal memory accepts the proven 8-byte allocation; a
    /// monotonically increasing epoch reuses it for every layer boundary.
    pub fn prepare_tp_graph_signals(&mut self, barriers: usize) -> HipResult<()> {
        if !matches!(self.devices.len(), 3 | 4)
            || !self
                .devices
                .iter()
                .all(|device| device.arch_caps.is_gfx1201())
        {
            return Err(HipError::new(
                0,
                "prepare_tp_graph_signals requires three or four gfx1201 devices",
            ));
        }
        if barriers == 0 {
            return Err(HipError::new(
                0,
                "prepare_tp_graph_signals requires at least one barrier",
            ));
        }
        if !self.tp_graph_signals.is_empty() {
            return if self.tp_graph_barrier_count == barriers {
                Ok(())
            } else {
                Err(HipError::new(
                    0,
                    "prepare_tp_graph_signals cannot resize a live signal tape",
                ))
            };
        }

        let bytes = std::mem::size_of::<u64>();
        let n = self.devices.len();
        let mut signals = Vec::with_capacity(n);
        for rank in 0..n {
            let gpu = &self.devices[rank];
            gpu.bind_thread()?;
            match gpu.hip.malloc_signal(bytes) {
                Ok(signal) => {
                    if let Err(error) = gpu.hip.memset(&signal, 0, bytes) {
                        let _ = gpu.hip.free(signal);
                        for (owner, allocated) in signals.drain(..).enumerate() {
                            let _ = self.devices[owner].hip.free(allocated);
                        }
                        return Err(error);
                    }
                    signals.push(signal);
                }
                Err(error) => {
                    for (owner, allocated) in signals.drain(..).enumerate() {
                        let _ = self.devices[owner].hip.free(allocated);
                    }
                    return Err(error);
                }
            }
        }
        self.tp_graph_signals = signals;
        self.tp_graph_barrier_count = barriers;
        self.tp_graph_capture_epoch = 0;
        Ok(())
    }

    /// Reset every captured producer epoch before launching any rank graph.
    pub fn reset_tp_graph_signals(&mut self) -> HipResult<()> {
        if self.tp_graph_signals.len() != self.devices.len() || self.tp_graph_barrier_count == 0 {
            return Err(HipError::new(0, "TP graph signal tape is not prepared"));
        }
        let bytes = std::mem::size_of::<u64>();
        for rank in 0..self.devices.len() {
            let gpu = &self.devices[rank];
            gpu.bind_thread()?;
            gpu.hip.memset(&self.tp_graph_signals[rank], 0, bytes)?;
        }
        Ok(())
    }

    /// Rewind the capture-time barrier cursor. The next 86 DS4 boundaries map
    /// deterministically to signal slots 0..85 in every rank graph.
    pub fn begin_tp_graph_signal_capture(&mut self) -> HipResult<()> {
        if self.tp_graph_signals.len() != self.devices.len() || self.tp_graph_barrier_count == 0 {
            return Err(HipError::new(0, "TP graph signal tape is not prepared"));
        }
        self.tp_graph_capture_epoch = 0;
        Ok(())
    }

    pub fn tp_graph_captured_signal_count(&self) -> usize {
        self.tp_graph_capture_epoch
    }

    pub fn tp_graph_signals_ready(&self, barriers: usize) -> bool {
        self.tp_graph_signals.len() == self.devices.len()
            && matches!(self.devices.len(), 3 | 4)
            && self.tp_graph_barrier_count == barriers
    }

    /// Free the exact-gated TP4 signal tape after every captured rank graph has
    /// been invalidated. Idempotent so failed-load and ordinary unload paths can
    /// share it.
    pub fn free_tp_graph_signals(&mut self) -> HipResult<()> {
        let signals = std::mem::take(&mut self.tp_graph_signals);
        let mut first_error = None;
        for (rank, signal) in signals.into_iter().enumerate() {
            if let Err(error) = self.devices[rank]
                .bind_thread()
                .and_then(|_| self.devices[rank].hip.free(signal))
            {
                first_error.get_or_insert(error);
            }
        }
        self.tp_graph_barrier_count = 0;
        self.tp_graph_capture_epoch = 0;
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn capture_tp_graph_barrier(&mut self) -> HipResult<()> {
        let epoch_index = self.tp_graph_capture_epoch;
        if epoch_index >= self.tp_graph_barrier_count {
            return Err(HipError::new(
                0,
                &format!(
                    "TP graph barrier epoch {epoch_index} exceeds prepared capacity {}",
                    self.tp_graph_barrier_count
                ),
            ));
        }
        let epoch = u32::try_from(epoch_index + 1)
            .map_err(|_| HipError::new(0, "TP graph barrier epoch exceeds u32"))?;
        let signals = &self.tp_graph_signals;
        let devices = &mut self.devices;
        let n = devices.len();
        for rank in 0..n {
            devices[rank].tp_graph_signal_store_gfx1201(&signals[rank], epoch)?;
        }
        for (destination, device) in devices.iter_mut().enumerate() {
            let peers: Vec<&DeviceBuffer> = (0..n)
                .filter(|&source| source != destination)
                .map(|source| &signals[source])
                .collect();
            if n == 3 {
                device.tp_graph_signal_wait2_gfx1201([peers[0], peers[1]], epoch)?;
            } else {
                device.tp_graph_signal_wait3_gfx1201([peers[0], peers[1], peers[2]], epoch)?;
            }
        }
        self.tp_graph_capture_epoch += 1;
        Ok(())
    }

    fn from_parts(
        devices: Vec<Gpu>,
        per_device: Vec<usize>,
        n_layers: usize,
        opts: &DeviceResolveOpts,
    ) -> HipResult<Self> {
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
            use_rccl: opts.tp_use_rccl.unwrap_or(true),
            devices,
            layer_to_device,
            band_starts,
            peer_access_enabled: false,
            output_device: n_devices - 1,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
            peer_ar_tmp: Vec::new(),
            peer_ar_tmp_bytes: 0,
            host_ar_tmp: Vec::new(),
            active_peer_lease: None,
            peer_lease_buffers: Vec::new(),
            peer_lease_next_id: 0,
            peer_lease_quarantined: false,
            rank_barrier_events: Vec::new(),
            tp_graph_signals: Vec::new(),
            tp_graph_barrier_count: 0,
            tp_graph_capture_epoch: 0,
        })
    }

    /// Lazily initialize RCCL communicators across all devices owned by
    /// this `Gpus`. Cached for process lifetime; subsequent calls are
    /// no-ops. A resolved `tp_use_rccl=false` short-circuits with a clear
    /// error so callers can fall through to a host-driven path (not
    /// yet implemented — Stage 2 follow-up).
    pub fn ensure_rccl(&mut self) -> HipResult<()> {
        if self.rccl_comms.is_some() {
            return Ok(());
        }
        if !self.use_rccl {
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

    /// All-reduce-sum of f32 buffers across a participating device group.
    /// `buffers[k]` must be a device pointer on `self.devices[group[k]]`
    /// holding `count` f32 elements; after this call, every group buffer
    /// holds the element-wise sum. In-place (send == recv).
    ///
    /// The RCCL communicator is owned by this entire `Gpus` set, so this
    /// implementation accepts only the full ordered group `0..n`. Use
    /// [`Self::all_reduce_sum_f32_peer`] for genuine subgroup reductions.
    /// Synchronization remains the caller's responsibility.
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
        let n = self.devices.len();
        if group.len() != n || !group.iter().copied().eq(0..n) {
            return Err(HipError::new(
                0,
                "all_reduce_sum_f32 (RCCL): sub-group reduction needs ncclCommSplit \
                 (Phase 5b); use all_reduce_sum_f32_peer for sub-groups.",
            ));
        }
        // Single-rank (TP=1) all-reduce is the identity and does not require
        // librccl.
        if n == 1 {
            return Ok(());
        }
        self.ensure_rccl()?;

        let rccl = self.rccl_comms.as_ref().expect("ensure_rccl populated it");
        rccl.group_start()
            .map_err(|e| HipError::new(0, &format!("ncclGroupStart: {e}")))?;
        for (rank, buf) in buffers.iter().enumerate() {
            let dev = &self.devices[rank];
            dev.bind_thread()?;
            let stream = dev.active_stream.as_ref().ok_or_else(|| {
                HipError::new(
                    0,
                    &format!(
                        "all_reduce_sum_f32: device {rank} has no active_stream — \
                         set `gpus.devices[{rank}].active_stream = Some(stream)` before calling.",
                    ),
                )
            })?;
            // SAFETY: `buf` is a live device buffer of `count` f32 on device
            // `rank`, and `stream` is that device's active stream.
            unsafe {
                rccl.all_reduce_sum_f32(
                    rank,
                    buf.as_ptr() as *const f32,
                    buf.as_ptr() as *mut f32,
                    count,
                    stream.raw_ptr(),
                )
            }
            .map_err(|e| HipError::new(0, &format!("ncclAllReduce rank={rank}: {e}")))?;
        }
        rccl.group_end()
            .map_err(|e| HipError::new(0, &format!("ncclGroupEnd: {e}")))?;
        Ok(())
    }

    /// Host-staged all-reduce-sum of f32 buffers across all ranks. No
    /// cross-device HIP copy or RCCL is used — this is the correctness-first
    /// fallback for mixed dense TP topologies where peer access is asymmetric
    /// (e.g. physical GPU2 unreachable). Each rank's `active_stream` is
    /// explicitly synchronized so the producer's partial is visible, then
    /// `memcpy_dtoh` downloads each partial into persistent pageable host
    /// staging (`host_ar_tmp`), an exact left-associated sum
    /// `rank0+rank1+...` is reduced into row 0, and the resulting bytes are
    /// synchronously `memcpy_htod`-uploaded to every rank buffer. On steady
    /// state after the maximum `count` has been seen, no allocation occurs:
    /// the outer vec is sized to `N` and each row is `resize`-grown only when
    /// `count` exceeds its current length.
    pub fn all_reduce_sum_f32_host(
        &mut self,
        buffers: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let n = self.devices.len();
        if buffers.len() != n {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_f32_host: buffers.len()={} != n_devices={n}",
                    buffers.len()
                ),
            ));
        }
        if n == 1 {
            return Ok(());
        }
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipError::new(0, "all_reduce_sum_f32_host: count overflow"))?;
        for (rank, buffer) in buffers.iter().enumerate() {
            if buffer.size() < bytes {
                return Err(HipError::new(
                    0,
                    &format!(
                        "all_reduce_sum_f32_host: rank {rank} buffer bytes {} < required {bytes}",
                        buffer.size()
                    ),
                ));
            }
        }
        // Persistent pageable host staging: grow outer to N, each row to at
        // least `count` f32. Steady state after max count does no alloc.
        if self.host_ar_tmp.len() != n {
            self.host_ar_tmp.resize_with(n, Vec::new);
        }
        for row in &mut self.host_ar_tmp {
            if row.len() < count {
                row.resize(count, 0.0);
            }
        }
        if bytes == 0 {
            return Ok(());
        }
        // D2H: synchronize producer stream, then download into row's prefix.
        // Preserve first error immediately — no partial-success claim.
        for (r, ((dev, buffer), row)) in self
            .devices
            .iter()
            .zip(buffers.iter())
            .zip(self.host_ar_tmp.iter_mut())
            .enumerate()
        {
            dev.bind_thread()?;
            let stream = dev.active_stream.as_ref().ok_or_else(|| {
                HipError::new(
                    0,
                    &format!(
                        "all_reduce_sum_f32_host: device {r} has no active_stream — \
                         set `gpus.devices[r].active_stream = Some(stream)` before calling.",
                    ),
                )
            })?;
            dev.hip.stream_synchronize(stream)?;
            // SAFETY: row.len() >= count ensured above, so first `count` f32
            // (bytes) lie within the Vec's initialized length. The derived
            // `[u8]` slice is exactly `bytes` and is bounded to that prefix.
            let dst: &mut [u8] = unsafe {
                let ptr = row.as_mut_ptr() as *mut u8;
                std::slice::from_raw_parts_mut(ptr, bytes)
            };
            dev.hip.memcpy_dtoh(dst, buffer)?;
        }
        // Exact left-associated reduction into row 0: rank0+rank1+...
        Self::host_reduce_rows(&mut self.host_ar_tmp, count);
        // H2D: synchronously upload row0's summed prefix to every rank.
        // SAFETY: row0.len() >= count, bytes bounded to first count f32.
        let src: &[u8] = unsafe {
            let ptr = self.host_ar_tmp[0].as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, bytes)
        };
        for (dev, buffer) in self.devices.iter().zip(buffers.iter()) {
            dev.bind_thread()?;
            dev.hip.memcpy_htod(buffer, src)?;
        }
        Ok(())
    }

    /// Pure left-associated host reduction for `all_reduce_sum_f32_host`.
    /// Sums rows 1..N element-wise into row 0 in rank order:
    /// `row0 = ((row0 + row1) + row2) + ...` over the first `count` f32.
    /// Tail beyond `count` is untouched (count-prefix behavior). No HIP.
    fn host_reduce_rows(rows: &mut [Vec<f32>], count: usize) {
        if rows.len() <= 1 || count == 0 {
            return;
        }
        // Clamp count to each row's actual length (callers guarantee len >= count).
        let count = count.min(rows[0].len());
        let (first, rest) = rows.split_at_mut(1);
        let acc = &mut first[0][..count];
        for other in rest.iter_mut() {
            let src = &other[..count];
            for i in 0..count {
                // Left-associated: acc accumulates in rank order.
                acc[i] += src[i];
            }
        }
    }

    /// Allocate unleased peer-rooted all-reduce scratch before enabling peer
    /// access. ROCm does not retroactively map allocations made after
    /// `hipDeviceEnablePeerAccess`, so callers that may use this scratch on
    /// peer-capable links must reserve their maximum reduction size first.
    pub fn reserve_peer_reduce_scratch(&mut self, bytes: usize) -> HipResult<()> {
        self.ensure_peer_ar_tmp(bytes)
    }

    /// Free unleased peer-rooted all-reduce scratch on owning devices.
    ///
    /// Idempotent: empty scratch zeroes `peer_ar_tmp_bytes` and returns `Ok`.
    /// Refuses while a unique peer lease is active or retained lease buffers
    /// remain.
    ///
    /// Safety order: bind + `device_synchronize` every owning device first so
    /// scratch is not in pending GPU work, then free. A bind/sync failure
    /// quarantines and returns the first error **without** freeing — residual
    /// rows and `peer_ar_tmp_bytes` stay intact for a later retry once the
    /// device is bindable/quiescent again. A free-phase failure also
    /// quarantines; any row that could not be taken remains nonempty and
    /// bytes stay nonzero while residual buffers exist.
    pub fn free_peer_reduce_scratch(&mut self) -> HipResult<()> {
        if self.active_peer_lease.is_some() || !self.peer_lease_buffers.is_empty() {
            return Err(HipError::new(
                0,
                "free_peer_reduce_scratch: peer scratch is leased — refuse free while a lease lives",
            ));
        }
        if self.peer_ar_tmp.is_empty() {
            self.peer_ar_tmp_bytes = 0;
            return Ok(());
        }

        // Phase 1: drain every owning device before any hipFree.
        let mut first_err: Option<HipError> = None;
        let n = self.peer_ar_tmp.len();
        for r in 0..n {
            if let Err(error) = self.devices[r].bind_thread() {
                first_err.get_or_insert(error);
                continue;
            }
            if let Err(error) = self.devices[r].hip.device_synchronize() {
                first_err.get_or_insert(error);
            }
        }
        if let Some(e) = first_err {
            self.peer_lease_quarantined = true;
            // Residual rows + bytes intentionally left for retry.
            return Err(e);
        }

        // Phase 2: free only after all owners are known quiescent.
        for r in 0..n {
            if let Err(error) = self.devices[r].bind_thread() {
                first_err.get_or_insert(error);
                continue;
            }
            for buffer in std::mem::take(&mut self.peer_ar_tmp[r]) {
                if let Err(error) = self.devices[r].hip.free(buffer) {
                    first_err.get_or_insert(error);
                }
            }
        }
        if let Some(e) = first_err {
            self.peer_lease_quarantined = true;
            if self.peer_ar_tmp.iter().all(|row| row.is_empty()) {
                self.peer_ar_tmp.clear();
                self.peer_ar_tmp_bytes = 0;
            }
            return Err(e);
        }
        self.peer_ar_tmp.clear();
        self.peer_ar_tmp_bytes = 0;
        self.peer_lease_quarantined = false;
        Ok(())
    }

    /// Ensure `peer_ar_tmp[r]` holds `n-1` buffers of at least `bytes` on each
    /// device. Lazily allocates; grows (freeing the old set) if `bytes` exceeds
    /// the current size. No-op for `n <= 1`. Fails while a unique peer lease
    /// is active or quarantined — the leased path owns the scratch exclusively.
    fn ensure_peer_ar_tmp(&mut self, bytes: usize) -> HipResult<()> {
        if self.active_peer_lease.is_some()
            || self.peer_lease_quarantined
            || !self.peer_lease_buffers.is_empty()
        {
            return Err(HipError::new(
                0,
                "ensure_peer_ar_tmp: peer scratch is leased — unleased reduce/resize is forbidden while a lease lives",
            ));
        }
        let n = self.devices.len();
        if n <= 1 {
            return Ok(());
        }
        if !self.peer_ar_tmp.is_empty() && self.peer_ar_tmp_bytes >= bytes {
            return Ok(());
        }
        // Free the old (too-small) set via the drain-before-free path so
        // hipFree never races pending GPU work on these buffers.
        if !self.peer_ar_tmp.is_empty() {
            self.free_peer_reduce_scratch()?;
        }

        let mut all = Vec::with_capacity(n);
        for r in 0..n {
            self.devices[r].bind_thread()?;
            let mut row = Vec::with_capacity(n - 1);
            for _ in 0..(n - 1) {
                match self.devices[r].hip.malloc(bytes) {
                    Ok(buf) => row.push(buf),
                    Err(e) => {
                        // Roll back everything allocated so far on its owning device.
                        let mut first_err: Option<HipError> = Some(e);
                        for buf in row {
                            if let Err(fe) = self.devices[r]
                                .bind_thread()
                                .and_then(|_| self.devices[r].hip.free(buf))
                            {
                                first_err.get_or_insert(fe);
                            }
                        }
                        for (rr, rrow) in all.into_iter().enumerate() {
                            let _ = self.devices[rr].bind_thread();
                            for buf in rrow {
                                if let Err(fe) = self.devices[rr].hip.free(buf) {
                                    first_err.get_or_insert(fe);
                                }
                            }
                            let _ = self.devices[rr].hip.device_synchronize();
                        }
                        let _ = self.devices[r].hip.device_synchronize();
                        if first_err.is_some() {
                            self.peer_lease_quarantined = true;
                        }
                        return Err(first_err.unwrap());
                    }
                }
            }
            all.push(row);
        }
        self.peer_ar_tmp = all;
        self.peer_ar_tmp_bytes = bytes;
        Ok(())
    }

    /// Acquire an exclusive peer-rooted scratch lease for `bytes` per reduction.
    ///
    /// Allocates exactly `N` rows of `N-1` buffers (3 per rank, 12 total for
    /// TP4) on their owning devices. Rejects a second concurrent owner and
    /// quarantined state. Partial allocation rolls back on owning devices,
    /// binds/synchronizes owners, attempts every free, and preserves the
    /// first error. The returned `PeerReduceScratchLease` is opaque,
    /// non-Clone/non-Copy, and must be passed to leased reduce/release.
    pub fn acquire_peer_reduce_scratch(
        &mut self,
        bytes: usize,
    ) -> HipResult<PeerReduceScratchLease> {
        if self.peer_lease_quarantined {
            return Err(HipError::new(
                0,
                "acquire_peer_reduce_scratch: scratch is quarantined after a prior free failure",
            ));
        }
        if self.active_peer_lease.is_some() {
            return Err(HipError::new(
                0,
                "acquire_peer_reduce_scratch: a lease is already active — second owner rejected",
            ));
        }
        if !self.peer_ar_tmp.is_empty() {
            return Err(HipError::new(
                0,
                "acquire_peer_reduce_scratch: unleased scratch is live — free it before acquiring a lease",
            ));
        }
        let n = self.devices.len();
        if n <= 1 {
            // Degenerate: still issue a lease but allocate nothing.
            let id = self
                .peer_lease_next_id
                .checked_add(1)
                .ok_or_else(|| HipError::new(0, "lease id overflow"))?;
            self.peer_lease_next_id = id;
            let lease = PeerReduceScratchLease {
                id,
                bytes,
                rank_count: n,
                _private: (),
            };
            self.active_peer_lease = Some(ActivePeerLease {
                id,
                bytes,
                rank_count: n,
            });
            self.peer_lease_buffers = Vec::new();
            return Ok(lease);
        }
        // Validate overflow for diagnostics via shared helper (pure).
        let _per_rank = peer_reduce_scratch_bytes_per_rank(n, bytes).ok_or_else(|| {
            HipError::new(
                0,
                "acquire_peer_reduce_scratch: overflow in per-rank projection",
            )
        })?;
        let mut all: Vec<Vec<DeviceBuffer>> = Vec::with_capacity(n);
        let mut first_err: Option<HipError> = None;
        for r in 0..n {
            if let Err(e) = self.devices[r].bind_thread() {
                first_err.get_or_insert(e);
                break;
            }
            let mut row = Vec::with_capacity(n - 1);
            for _ in 0..(n - 1) {
                match self.devices[r].hip.malloc(bytes) {
                    Ok(buf) => row.push(buf),
                    Err(e) => {
                        first_err.get_or_insert(e);
                        break;
                    }
                }
            }
            if first_err.is_some() {
                all.push(row);
                break;
            }
            all.push(row);
            if all.len() != r + 1 {
                break;
            }
        }
        if let Some(e) = first_err {
            // Roll back every allocated row on its owning device, bind/sync, preserve first error.
            let mut rollback_first: Option<HipError> = Some(e);
            for (rr, rrow) in all.into_iter().enumerate() {
                if let Err(be) = self.devices[rr].bind_thread() {
                    rollback_first.get_or_insert(be);
                    continue;
                }
                for buf in rrow {
                    if let Err(fe) = self.devices[rr].hip.free(buf) {
                        rollback_first.get_or_insert(fe);
                    }
                }
                if let Err(se) = self.devices[rr].hip.device_synchronize() {
                    rollback_first.get_or_insert(se);
                }
            }
            self.peer_lease_quarantined = true;
            return Err(rollback_first.unwrap());
        }
        if all.len() != n {
            // Defensive: ensure we built N rows.
            let mut rb_first: Option<HipError> = None;
            for (rr, rrow) in all.into_iter().enumerate() {
                let _ = self.devices[rr].bind_thread();
                for buf in rrow {
                    if let Err(fe) = self.devices[rr].hip.free(buf) {
                        rb_first.get_or_insert(fe);
                    }
                }
                let _ = self.devices[rr].hip.device_synchronize();
            }
            self.peer_lease_quarantined = true;
            return Err(rb_first.unwrap_or_else(|| {
                HipError::new(0, "acquire_peer_reduce_scratch: incomplete allocation")
            }));
        }
        let id = self
            .peer_lease_next_id
            .checked_add(1)
            .ok_or_else(|| HipError::new(0, "lease id overflow"))?;
        self.peer_lease_next_id = id;
        self.active_peer_lease = Some(ActivePeerLease {
            id,
            bytes,
            rank_count: n,
        });
        self.peer_lease_buffers = all;
        Ok(PeerReduceScratchLease {
            id,
            bytes,
            rank_count: n,
            _private: (),
        })
    }

    /// Release an active peer lease, freeing its scratch on owning devices.
    ///
    /// Binds and synchronizes every owning device, attempts every free,
    /// preserves the first error, and clears the active lease only after
    /// complete success. A failed release quarantines the scratch so no
    /// future owner can use a partially freed set.
    pub fn release_peer_reduce_scratch(&mut self, lease: &PeerReduceScratchLease) -> HipResult<()> {
        let active = self
            .active_peer_lease
            .as_ref()
            .ok_or_else(|| HipError::new(0, "release_peer_reduce_scratch: no active lease"))?;
        if active.id != lease.id {
            return Err(HipError::new(
                0,
                &format!(
                    "release_peer_reduce_scratch: lease id {} != active {}",
                    lease.id, active.id
                ),
            ));
        }
        if active.bytes != lease.bytes || active.rank_count != lease.rank_count {
            return Err(HipError::new(
                0,
                "release_peer_reduce_scratch: lease bytes/rank_count mismatch",
            ));
        }
        if lease.rank_count != self.devices.len() {
            return Err(HipError::new(
                0,
                &format!(
                    "release_peer_reduce_scratch: lease rank_count {} != n_devices {}",
                    lease.rank_count,
                    self.devices.len()
                ),
            ));
        }
        let buffers = std::mem::take(&mut self.peer_lease_buffers);
        let mut first_err: Option<HipError> = None;
        for (r, row) in buffers.into_iter().enumerate() {
            if let Err(e) = self.devices[r].bind_thread() {
                first_err.get_or_insert(e);
                continue;
            }
            for buf in row {
                if let Err(e) = self.devices[r].hip.free(buf) {
                    first_err.get_or_insert(e);
                }
            }
            if let Err(e) = self.devices[r].hip.device_synchronize() {
                first_err.get_or_insert(e);
            }
        }
        if let Some(e) = first_err {
            self.peer_lease_quarantined = true;
            // Keep the active lease so no future acquire can succeed without quarantine clear.
            // Buffers already taken; leave empty to prevent use.
            return Err(e);
        }
        self.active_peer_lease = None;
        self.peer_lease_quarantined = false;
        Ok(())
    }

    /// All-reduce-sum of f32 buffers across a participating device group via
    /// direct peer copies and local adds, bypassing RCCL.
    ///
    /// `group` lists global device IDs and `buffers[k]` belongs to
    /// `self.devices[group[k]]`. Unlike the RCCL path this is genuinely
    /// subgroup-capable, which is required for composed TP×EP meshes.
    pub fn all_reduce_sum_f32_peer(
        &mut self,
        group: &[usize],
        buffers: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        if self.active_peer_lease.is_some()
            || self.peer_lease_quarantined
            || !self.peer_lease_buffers.is_empty()
        {
            return Err(HipError::new(
                0,
                "all_reduce_sum_f32_peer: peer scratch is leased — use leased API or release lease",
            ));
        }
        let g = group.len();
        let n = self.devices.len();
        if buffers.len() != g {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_f32_peer: buffers.len()={} != group.len()={g}",
                    buffers.len()
                ),
            ));
        }
        for (index, &device) in group.iter().enumerate() {
            if device >= n || group[..index].contains(&device) {
                return Err(HipError::new(
                    0,
                    &format!(
                        "all_reduce_sum_f32_peer: group contains invalid or duplicate device {device}"
                    ),
                ));
            }
        }
        if g <= 1 {
            return Ok(());
        }
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipError::new(0, "all_reduce_sum_f32_peer: count overflow"))?;
        self.ensure_peer_ar_tmp(bytes)?;

        // Phase 1: read every peer's ORIGINAL buffer into a local temp.
        let mut events = Vec::with_capacity(g * (g - 1));
        for (k, &device_k) in group.iter().enumerate() {
            let mut slot = 0;
            for (m, &device_m) in group.iter().enumerate() {
                if m == k {
                    continue;
                }
                let event = self.boundary_copy(
                    device_m,
                    device_k,
                    buffers[m],
                    &self.peer_ar_tmp[device_k][slot],
                    bytes,
                )?;
                events.push(event);
                slot += 1;
            }
        }
        for event in events {
            self.wait_boundary(event)?;
        }

        // Phase 2: add peer temps into each rank's buffer.
        for (k, &device_k) in group.iter().enumerate() {
            let dst = GpuTensor {
                buf: unsafe { buffers[k].alias() },
                shape: vec![count],
                dtype: DType::F32,
            };
            let srcs: Vec<GpuTensor> = (0..g - 1)
                .map(|slot| GpuTensor {
                    buf: unsafe { self.peer_ar_tmp[device_k][slot].alias() },
                    shape: vec![count],
                    dtype: DType::F32,
                })
                .collect();
            self.devices[device_k].bind_thread()?;
            for src in &srcs {
                self.devices[device_k].add_inplace_f32(&dst, src)?;
            }
        }
        Ok(())
    }

    /// Deterministic rooted peer all-reduce for batched TP activations.
    ///
    /// Every rank observes the exact same left-associated sum
    /// `rank0 + rank1 + ... + rankN`: peer inputs are first copied into
    /// rank 0 scratch, rank 0 accumulates them in rank order, and the result
    /// is broadcast back to every peer. This both avoids rank-dependent
    /// floating-point order and cuts peer traffic from `N * (N - 1)` copies
    /// to `2 * (N - 1)` copies. Buffers are updated in place.
    pub fn all_reduce_sum_f32_peer_rooted(
        &mut self,
        buffers: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        self.reduce_sum_f32_peer_rooted_impl(buffers, None, count, "all_reduce_sum_f32_peer_rooted")
    }

    /// Deterministically reduce rank-local partials, add the completed sum to
    /// rank 0's residual, then broadcast that completed residual to every rank.
    ///
    /// The arithmetic order is `(((partial0 + partial1) + ...) + residual0)`.
    /// Peer residuals are overwritten by the root result, avoiding one local
    /// residual-add kernel on every non-root rank.
    pub fn all_reduce_sum_f32_peer_rooted_add(
        &mut self,
        partials: &[&DeviceBuffer],
        residuals: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        self.reduce_sum_f32_peer_rooted_impl(
            partials,
            Some(residuals),
            count,
            "all_reduce_sum_f32_peer_rooted_add",
        )
    }

    fn reduce_sum_f32_peer_rooted_impl(
        &mut self,
        partials: &[&DeviceBuffer],
        residuals: Option<&[&DeviceBuffer]>,
        count: usize,
        operation: &'static str,
    ) -> HipResult<()> {
        if self.active_peer_lease.is_some()
            || self.peer_lease_quarantined
            || !self.peer_lease_buffers.is_empty()
        {
            return Err(HipError::new(
                0,
                &format!("{operation}: peer scratch is leased — use the leased reduction API"),
            ));
        }
        let n = self.devices.len();
        if partials.len() != n {
            return Err(HipError::new(
                0,
                &format!(
                    "{operation}: partials.len()={} != n_devices={n}",
                    partials.len()
                ),
            ));
        }
        if let Some(outputs) = residuals {
            if outputs.len() != n {
                return Err(HipError::new(
                    0,
                    &format!(
                        "{operation}: residuals.len()={} != n_devices={n}",
                        outputs.len()
                    ),
                ));
            }
        }
        if count == 0 {
            return Ok(());
        }
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipError::new(0, &format!("{operation}: byte count overflow")))?;
        for (rank, buffer) in partials.iter().enumerate() {
            if buffer.size() < bytes {
                return Err(HipError::new(
                    0,
                    &format!(
                        "{operation}: partial rank {rank} has {} bytes, needs {bytes}",
                        buffer.size()
                    ),
                ));
            }
        }
        if let Some(outputs) = residuals {
            for (rank, buffer) in outputs.iter().enumerate() {
                if buffer.size() < bytes {
                    return Err(HipError::new(
                        0,
                        &format!(
                            "{operation}: residual rank {rank} has {} bytes, needs {bytes}",
                            buffer.size()
                        ),
                    ));
                }
            }
        }
        if n == 1 {
            if let Some(outputs) = residuals {
                let partial = GpuTensor {
                    buf: unsafe { partials[0].alias() },
                    shape: vec![count],
                    dtype: DType::F32,
                };
                let residual = GpuTensor {
                    buf: unsafe { outputs[0].alias() },
                    shape: vec![count],
                    dtype: DType::F32,
                };
                self.devices[0].bind_thread()?;
                self.devices[0].add_f32(&residual, &partial, &residual)?;
            }
            return Ok(());
        }
        self.ensure_peer_ar_tmp(bytes)?;

        let mut gather_events = Vec::with_capacity(n - 1);
        for (rank, partial) in partials.iter().enumerate().take(n).skip(1) {
            gather_events.push(self.boundary_copy(
                rank,
                0,
                partial,
                &self.peer_ar_tmp[0][rank - 1],
                bytes,
            )?);
        }
        for event in gather_events {
            self.wait_boundary(event)?;
        }

        let root_partial = GpuTensor {
            buf: unsafe { partials[0].alias() },
            shape: vec![count],
            dtype: DType::F32,
        };
        self.devices[0].bind_thread()?;
        for slot in 0..n - 1 {
            let peer = GpuTensor {
                buf: unsafe { self.peer_ar_tmp[0][slot].alias() },
                shape: vec![count],
                dtype: DType::F32,
            };
            self.devices[0].add_inplace_f32(&root_partial, &peer)?;
        }

        let broadcast = if let Some(outputs) = residuals {
            let root_residual = GpuTensor {
                buf: unsafe { outputs[0].alias() },
                shape: vec![count],
                dtype: DType::F32,
            };
            self.devices[0].add_f32(&root_residual, &root_partial, &root_residual)?;
            outputs
        } else {
            partials
        };
        let mut broadcast_events = Vec::with_capacity(n - 1);
        for rank in 1..n {
            broadcast_events.push(self.boundary_copy(
                0,
                rank,
                broadcast[0],
                broadcast[rank],
                bytes,
            )?);
        }
        for event in broadcast_events {
            self.wait_boundary(event)?;
        }
        Ok(())
    }

    /// Leased variant of the deterministic rooted peer all-reduce. Uses the
    /// scratch allocated under `lease` and never allocates or grows. Validates
    /// lease identity, rank count, `count*4 <= lease.bytes`, row lengths and
    /// capacities. The rooted order is exactly `(((rank0 + rank1)+rank2)+rank3)`.
    pub fn all_reduce_sum_f32_peer_rooted_leased(
        &mut self,
        lease: &PeerReduceScratchLease,
        buffers: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let n = self.devices.len();
        let active = self.active_peer_lease.as_ref().ok_or_else(|| {
            HipError::new(0, "all_reduce_sum_f32_peer_rooted_leased: no active lease")
        })?;
        if active.id != lease.id {
            return Err(HipError::new(
                0,
                &format!(
                    "leased reduce: lease id {} != active {}",
                    lease.id, active.id
                ),
            ));
        }
        if active.bytes != lease.bytes || active.rank_count != lease.rank_count {
            return Err(HipError::new(
                0,
                "leased reduce: lease bytes/rank_count mismatch vs active record",
            ));
        }
        if lease.rank_count != n {
            return Err(HipError::new(
                0,
                &format!(
                    "leased reduce: lease rank_count {} != n_devices {}",
                    lease.rank_count, n
                ),
            ));
        }
        if buffers.len() != n {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_f32_peer_rooted_leased: buffers.len()={} != n_devices={n}",
                    buffers.len()
                ),
            ));
        }
        if n == 1 {
            return Ok(());
        }
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| HipError::new(0, "leased reduce: count overflow"))?;
        if bytes > lease.bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "leased reduce: count*4 {} > lease.bytes {}",
                    bytes, lease.bytes
                ),
            ));
        }
        if bytes > active.bytes {
            return Err(HipError::new(
                0,
                "leased reduce: count*4 exceeds active lease bytes",
            ));
        }
        if self.peer_lease_buffers.len() != n {
            return Err(HipError::new(
                0,
                "leased reduce: peer_lease_buffers row count mismatch",
            ));
        }
        for (r, row) in self.peer_lease_buffers.iter().enumerate() {
            if row.len() != n - 1 {
                return Err(HipError::new(
                    0,
                    &format!("leased reduce: row {r} len {} != N-1 {}", row.len(), n - 1),
                ));
            }
            for buf in row {
                if buf.size() < bytes {
                    return Err(HipError::new(
                        0,
                        "leased reduce: scratch buffer too small for count",
                    ));
                }
            }
        }
        if self.peer_lease_quarantined {
            return Err(HipError::new(0, "leased reduce: scratch is quarantined"));
        }
        // Gather N-1 peers into rank-0 lease scratch, never allocating.
        let mut gather_events = Vec::with_capacity(n - 1);
        for (rank, buffer) in buffers.iter().enumerate().take(n).skip(1) {
            gather_events.push(self.boundary_copy(
                rank,
                0,
                buffer,
                &self.peer_lease_buffers[0][rank - 1],
                bytes,
            )?);
        }
        for event in gather_events {
            self.wait_boundary(event)?;
        }
        let root = GpuTensor {
            buf: unsafe { buffers[0].alias() },
            shape: vec![count],
            dtype: DType::F32,
        };
        self.devices[0].bind_thread()?;
        for slot in 0..n - 1 {
            let peer = GpuTensor {
                buf: unsafe { self.peer_lease_buffers[0][slot].alias() },
                shape: vec![count],
                dtype: DType::F32,
            };
            self.devices[0].add_inplace_f32(&root, &peer)?;
        }
        let mut broadcast_events = Vec::with_capacity(n - 1);
        for rank in 1..n {
            broadcast_events.push(self.boundary_copy(0, rank, buffers[0], buffers[rank], bytes)?);
        }
        for event in broadcast_events {
            self.wait_boundary(event)?;
        }
        Ok(())
    }
}

/// Pure-TP PP band metadata: length `tp_size`, rank 0 owns every layer and
/// ranks ≥1 hold empty bands (`[0, n_layers, n_layers, …]`).
fn tp_band_starts(tp_size: usize, n_layers: usize) -> Vec<usize> {
    (0..tp_size)
        .map(|rank| if rank == 0 { 0 } else { n_layers })
        .collect()
}

fn uniform_split_counts(n_devices: usize, n_layers: usize) -> Vec<usize> {
    let base = n_layers / n_devices;
    let rem = n_layers % n_devices;
    (0..n_devices)
        .map(|i| base + if i < rem { 1 } else { 0 })
        .collect()
}

/// Resolve logical device IDs from the already-resolved `hardware.devices`
/// value, or use the first `n_devices` visible IDs. The supplied list is
/// post-visibility logical HIP IDs and is consumed without physical-ID
/// remapping.
fn resolve_device_ids(n_devices: usize, opts: &DeviceResolveOpts) -> HipResult<Vec<i32>> {
    if let Some(value) = &opts.devices {
        let parsed = value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::parse::<i32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| HipError::new(0, &format!("hardware.devices parse: {error}")))?;
        if parsed.len() < n_devices {
            return Err(HipError::new(
                0,
                &format!(
                    "hardware.devices exposes {} ids but n_devices = {n_devices}",
                    parsed.len()
                ),
            ));
        }
        return Ok(parsed[..n_devices].to_vec());
    }
    (0..n_devices)
        .map(|id| {
            i32::try_from(id)
                .map_err(|_| HipError::new(0, "hardware.devices logical ID exceeds HIP range"))
        })
        .collect()
}

fn construct_devices(ids: &[i32]) -> HipResult<Vec<Gpu>> {
    let mut devices = Vec::with_capacity(ids.len());
    for &id in ids {
        devices.push(Gpu::init_with_device(id)?);
    }
    Ok(devices)
}

fn preflight_vram_with_opts(
    devices: &[Gpu],
    check_vram_delta: bool,
    opts: &DeviceResolveOpts,
) -> HipResult<()> {
    if devices.is_empty() {
        return Ok(());
    }
    let arch0 = devices[0].arch.clone();
    let mut frees = Vec::with_capacity(devices.len());
    for device in devices {
        if device.arch != arch0 {
            if opts.allow_mixed_arch {
                eprintln!(
                    "preflight_vram: mixed-arch detected — dev 0 is {arch0}, dev {} is {}. \
                     Proceeding because HIPFIRE_ALLOW_MIXED_ARCH=1. \
                     Per-arch JIT cache will be populated on first run; boundary_copy uses \
                     hipMemcpyPeer / hipMemcpyPeerAsync which fall through to host-staging \
                     if peer access is unsupported by the pair (correctness holds either way).",
                    device.device_id, device.arch,
                );
            } else {
                return Err(HipError::new(
                    0,
                    &format!(
                        "preflight_vram: arch mismatch — dev 0 is {arch0}, dev {} is {}. \
                         Mixed-arch is not supported by default; set HIPFIRE_ALLOW_MIXED_ARCH=1 to override.",
                        device.device_id, device.arch,
                    ),
                ));
            }
        }
        device.bind_thread()?;
        let (free, _total) = device.hip.get_vram_info()?;
        frees.push(free);
    }
    if !check_vram_delta {
        return Ok(());
    }
    let max_free = *frees.iter().max().unwrap();
    let min_free = *frees.iter().min().unwrap();
    let delta_gb = (max_free - min_free) as f64 / 1e9;
    let tolerance_gb = opts
        .uniform_vram_tolerance_gb
        .map(|value| value as f64)
        .unwrap_or(DEFAULT_VRAM_TOLERANCE_GB);
    if delta_gb > tolerance_gb {
        return Err(HipError::new(
            0,
            &format!(
                "preflight_vram: VRAM delta {:.1} GiB exceeds tolerance {:.1} GiB. \
                 Override via HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB or use init_layers().",
                delta_gb, tolerance_gb,
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_split_basic() {
        assert_eq!(uniform_split_counts(2, 24), vec![12, 12]);
        assert_eq!(uniform_split_counts(2, 25), vec![13, 12]);
        assert_eq!(uniform_split_counts(3, 64), vec![22, 21, 21]);
        assert_eq!(uniform_split_counts(4, 7), vec![2, 2, 2, 1]);
    }

    #[test]
    fn tp_band_starts_length_and_values() {
        // TP=1: single entry, rank 0 owns all layers.
        assert_eq!(tp_band_starts(1, 40), vec![0]);
        // TP=4: exactly 4 entries — rank 0 owns [0, n_layers), others empty.
        // Would fail against the old `0..=tp_size` (5-entry) construction.
        assert_eq!(tp_band_starts(4, 40), vec![0, 40, 40, 40]);
        assert_eq!(tp_band_starts(4, 40).len(), 4);
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

    #[test]
    fn peer_rooted_projection_n4_buffers_per_device_and_total() {
        // N=4 → N-1 = 3 scratch buffers per device, 12 total.
        let requested = 4096usize;
        let per = peer_reduce_scratch_bytes_per_rank(4, requested).expect("N=4 per-rank");
        let total = peer_reduce_scratch_total_bytes(4, requested).expect("N=4 total");
        assert_eq!(per, 3 * requested, "3 scratch buffers per device");
        assert_eq!(total, 12 * requested, "12 scratch buffers total");
        // Buffer counts implied by the byte projection (one buffer = requested_bytes).
        assert_eq!(per / requested, 3);
        assert_eq!(total / requested, 12);
        assert_eq!(total, 4 * per);
    }

    #[test]
    fn peer_rooted_projection_requested_bytes_multiplication_exact() {
        for &requested in &[0usize, 1, 4, 64, 1024, 4096, 1 << 20] {
            for n in 1usize..=8 {
                let per = peer_reduce_scratch_bytes_per_rank(n, requested)
                    .unwrap_or_else(|| panic!("per-rank None for n={n} req={requested}"));
                let total = peer_reduce_scratch_total_bytes(n, requested)
                    .unwrap_or_else(|| panic!("total None for n={n} req={requested}"));
                assert_eq!(per, (n - 1).checked_mul(requested).unwrap());
                assert_eq!(total, n.checked_mul(per).unwrap());
                assert_eq!(
                    total,
                    n.checked_mul(n - 1)
                        .unwrap()
                        .checked_mul(requested)
                        .unwrap()
                );
            }
        }
        // Zero ranks is rejected (not a projection).
        assert_eq!(peer_reduce_scratch_bytes_per_rank(0, 64), None);
        assert_eq!(peer_reduce_scratch_total_bytes(0, 64), None);
    }

    #[test]
    fn peer_rooted_projection_checked_overflow_returns_error() {
        // per_rank = (n-1) * requested overflows → None.
        let huge = usize::MAX / 2 + 1;
        assert_eq!(
            peer_reduce_scratch_bytes_per_rank(4, huge),
            None,
            "3 * huge must overflow"
        );
        assert_eq!(
            peer_reduce_scratch_total_bytes(4, huge),
            None,
            "total inherits per-rank overflow"
        );

        // per_rank fits but total = n * per_rank overflows → None.
        // For n=4: per = 3 * req; total = 4 * 3 * req = 12 * req.
        // Choose req so 3*req fits but 12*req overflows.
        let req = (usize::MAX / 3).saturating_sub(0);
        // Ensure 3*req is Some (fits) when possible; if 3*req already overflows, still None.
        if let Some(per) = peer_reduce_scratch_bytes_per_rank(4, req) {
            assert!(
                4usize.checked_mul(per).is_none()
                    || peer_reduce_scratch_total_bytes(4, req).is_some(),
                "when total fits, helper must agree"
            );
            if 4usize.checked_mul(per).is_none() {
                assert_eq!(peer_reduce_scratch_total_bytes(4, req), None);
            }
        }

        // Direct total overflow: pick n and req where (n-1)*req fits in usize but n*(n-1)*req does not.
        // n=3 → per = 2*req; total = 3*2*req = 6*req.
        let req2 = usize::MAX / 4; // 2*req2 fits; 6*req2 may overflow
        if let Some(per2) = peer_reduce_scratch_bytes_per_rank(3, req2) {
            if 3usize.checked_mul(per2).is_none() {
                assert_eq!(peer_reduce_scratch_total_bytes(3, req2), None);
            }
        }

        // Maximum multiply that still overflows for N=4 per-rank path.
        assert_eq!(peer_reduce_scratch_bytes_per_rank(4, usize::MAX), None);
        assert_eq!(peer_reduce_scratch_total_bytes(4, usize::MAX), None);
        assert_eq!(peer_reduce_scratch_bytes_per_rank(usize::MAX, 2), None);
        assert_eq!(peer_reduce_scratch_total_bytes(usize::MAX, 2), None);
    }

    #[test]
    fn host_reduce_rows_left_associated_and_count_prefix() {
        // Rank-ordered left-associated sum: row0 = ((row0+row1)+row2)+...
        let mut rows = vec![
            vec![1.0_f32, 2.0, 3.0, 99.0],
            vec![10.0, 20.0, 30.0, 88.0],
            vec![100.0, 200.0, 300.0, 77.0],
        ];
        Gpus::host_reduce_rows(&mut rows, 3);
        assert_eq!(rows[0][0], 111.0);
        assert_eq!(rows[0][1], 222.0);
        assert_eq!(rows[0][2], 333.0);
        // Count-prefix: tail beyond count untouched, and other rows unchanged.
        assert_eq!(rows[0][3], 99.0);
        assert_eq!(rows[1], vec![10.0, 20.0, 30.0, 88.0]);
        assert_eq!(rows[2], vec![100.0, 200.0, 300.0, 77.0]);

        // Count smaller than row length: only first `count` elements summed.
        let mut rows2 = vec![vec![1.0_f32, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        Gpus::host_reduce_rows(&mut rows2, 2);
        assert_eq!(rows2[0][0], 11.0);
        assert_eq!(rows2[0][1], 22.0);
        assert_eq!(rows2[0][2], 3.0);
        assert_eq!(rows2[0][3], 4.0);

        // Count zero is no-op.
        let mut rows3 = vec![vec![5.0_f32, 6.0], vec![7.0, 8.0]];
        Gpus::host_reduce_rows(&mut rows3, 0);
        assert_eq!(rows3[0], vec![5.0, 6.0]);
    }
    #[test]
    fn resolver_consumes_post_visibility_logical_ids_without_aliasing() {
        let opts = DeviceResolveOpts {
            devices: Some("0,1".into()),
            ..DeviceResolveOpts::default()
        };
        assert_eq!(resolve_device_ids(2, &opts).unwrap(), vec![0, 1]);
    }
}
