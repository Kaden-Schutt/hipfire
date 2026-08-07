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
    /// One process-lifetime dependency event per rank for peer-consumer
    /// collectives. Re-recording avoids 86 create/destroy pairs per DS4 token.
    rank_barrier_events: Vec<Event>,
    /// One contiguous system-visible signal array per rank for the exact-gated
    /// gfx1201 TP4 graph route. Each layer consumes two slots (attention + FFN).
    tp4_graph_signals: Vec<DeviceBuffer>,
    tp4_graph_signal_slots: usize,
    tp4_graph_capture_slot: usize,
}

const DEFAULT_VRAM_TOLERANCE_GB: f64 = 2.0;

impl Gpus {
    /// Construct `n_devices` `Gpu` instances bound to logical IDs derived from
    /// synchronized `hardware.devices` visibility (or the first N visible if
    /// unset). Layers are split
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
            rank_barrier_events: Vec::new(),
            tp4_graph_signals: Vec::new(),
            tp4_graph_signal_slots: 0,
            tp4_graph_capture_slot: 0,
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
            rank_barrier_events: Vec::new(),
            tp4_graph_signals: Vec::new(),
            tp4_graph_signal_slots: 0,
            tp4_graph_capture_slot: 0,
        })
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
        if self.tp4_graph_signals.len() == 4
            && self.devices.len() == 4
            && self.devices.iter().all(|device| device.graphs.capture_mode)
        {
            return self.capture_tp4_graph_barrier();
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

    /// Allocate the fixed gfx1201 TP4 graph barrier tape before peer access is
    /// enabled. Signal memory is system-visible, and one contiguous allocation
    /// per rank lets replay clear all layer epochs with four host calls.
    pub fn prepare_tp4_graph_signals(&mut self, slots: usize) -> HipResult<()> {
        if self.devices.len() != 4
            || !self
                .devices
                .iter()
                .all(|device| device.arch_caps.is_gfx1201())
        {
            return Err(HipError::new(
                0,
                "prepare_tp4_graph_signals requires exactly four gfx1201 devices",
            ));
        }
        if slots == 0 {
            return Err(HipError::new(
                0,
                "prepare_tp4_graph_signals requires at least one slot",
            ));
        }
        if !self.tp4_graph_signals.is_empty() {
            return if self.tp4_graph_signal_slots == slots {
                Ok(())
            } else {
                Err(HipError::new(
                    0,
                    "prepare_tp4_graph_signals cannot resize a live signal tape",
                ))
            };
        }

        let bytes = slots
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| HipError::new(0, "TP4 graph signal byte size overflow"))?;
        let mut signals = Vec::with_capacity(4);
        for rank in 0..4 {
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
        self.tp4_graph_signals = signals;
        self.tp4_graph_signal_slots = slots;
        self.tp4_graph_capture_slot = 0;
        Ok(())
    }

    /// Reset every captured producer epoch before launching any rank graph.
    pub fn reset_tp4_graph_signals(&mut self) -> HipResult<()> {
        if self.tp4_graph_signals.len() != 4 || self.tp4_graph_signal_slots == 0 {
            return Err(HipError::new(0, "TP4 graph signal tape is not prepared"));
        }
        let bytes = self.tp4_graph_signal_slots * std::mem::size_of::<u32>();
        for rank in 0..4 {
            let gpu = &self.devices[rank];
            gpu.bind_thread()?;
            gpu.hip.memset(&self.tp4_graph_signals[rank], 0, bytes)?;
        }
        Ok(())
    }

    /// Rewind the capture-time barrier cursor. The next 86 DS4 boundaries map
    /// deterministically to signal slots 0..85 in every rank graph.
    pub fn begin_tp4_graph_signal_capture(&mut self) -> HipResult<()> {
        if self.tp4_graph_signals.len() != 4 || self.tp4_graph_signal_slots == 0 {
            return Err(HipError::new(0, "TP4 graph signal tape is not prepared"));
        }
        self.tp4_graph_capture_slot = 0;
        Ok(())
    }

    pub fn tp4_graph_captured_signal_count(&self) -> usize {
        self.tp4_graph_capture_slot
    }

    pub fn tp4_graph_signals_ready(&self, slots: usize) -> bool {
        self.tp4_graph_signals.len() == 4 && self.tp4_graph_signal_slots == slots
    }

    /// Free the exact-gated TP4 signal tape after every captured rank graph has
    /// been invalidated. Idempotent so failed-load and ordinary unload paths can
    /// share it.
    pub fn free_tp4_graph_signals(&mut self) -> HipResult<()> {
        let signals = std::mem::take(&mut self.tp4_graph_signals);
        let mut first_error = None;
        for (rank, signal) in signals.into_iter().enumerate() {
            if let Err(error) = self.devices[rank]
                .bind_thread()
                .and_then(|_| self.devices[rank].hip.free(signal))
            {
                first_error.get_or_insert(error);
            }
        }
        self.tp4_graph_signal_slots = 0;
        self.tp4_graph_capture_slot = 0;
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn capture_tp4_graph_barrier(&mut self) -> HipResult<()> {
        let slot = self.tp4_graph_capture_slot;
        if slot >= self.tp4_graph_signal_slots {
            return Err(HipError::new(
                0,
                &format!(
                    "TP4 graph barrier slot {slot} exceeds prepared capacity {}",
                    self.tp4_graph_signal_slots
                ),
            ));
        }
        let slot_i32 = i32::try_from(slot)
            .map_err(|_| HipError::new(0, "TP4 graph barrier slot exceeds i32"))?;
        let signals = &self.tp4_graph_signals;
        let devices = &mut self.devices;
        for rank in 0..4 {
            devices[rank].tp4_graph_signal_store_gfx1201(&signals[rank], slot_i32)?;
        }
        for destination in 0..4 {
            let peers: Vec<&DeviceBuffer> = (0..4)
                .filter(|&source| source != destination)
                .map(|source| &signals[source])
                .collect();
            devices[destination]
                .tp4_graph_signal_wait3_gfx1201([peers[0], peers[1], peers[2]], slot_i32)?;
        }
        self.tp4_graph_capture_slot += 1;
        Ok(())
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
            rank_barrier_events: Vec::new(),
            tp4_graph_signals: Vec::new(),
            tp4_graph_signal_slots: 0,
            tp4_graph_capture_slot: 0,
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
        if matches!(crate::config::get().tp_use_rccl, Some(false)) {
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
    pub fn all_reduce_sum_f32(&mut self, buffers: &[&DeviceBuffer], count: usize) -> HipResult<()> {
        if buffers.len() != self.devices.len() {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_f32: buffers.len()={} != n_devices={}",
                    buffers.len(),
                    self.devices.len()
                ),
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
            // `r`, and `stream` is that device's active stream.
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
    pub fn all_reduce_sum_f32_peer(
        &mut self,
        buffers: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let n = self.devices.len();
        if buffers.len() != n {
            return Err(HipError::new(
                0,
                &format!(
                    "all_reduce_sum_f32_peer: buffers.len()={} != n_devices={n}",
                    buffers.len()
                ),
            ));
        }
        if n == 1 {
            return Ok(());
        }
        let bytes = count * 4;
        self.ensure_peer_ar_tmp(bytes)?;

        // Phase 1: read every peer's ORIGINAL buffer into a local temp.
        let mut evts = Vec::with_capacity(n * (n - 1));
        for r in 0..n {
            let mut slot = 0usize;
            for j in 0..n {
                if j == r {
                    continue;
                }
                let evt =
                    self.boundary_copy(j, r, buffers[j], &self.peer_ar_tmp[r][slot], bytes)?;
                evts.push(evt);
                slot += 1;
            }
        }
        for evt in evts {
            self.wait_boundary(evt)?;
        }

        // Phase 2: add the peer temps into each rank's buffer.
        for r in 0..n {
            let dst = GpuTensor {
                buf: unsafe { buffers[r].alias() },
                shape: vec![count],
                dtype: DType::F32,
            };
            let srcs: Vec<GpuTensor> = (0..n - 1)
                .map(|slot| GpuTensor {
                    buf: unsafe { self.peer_ar_tmp[r][slot].alias() },
                    shape: vec![count],
                    dtype: DType::F32,
                })
                .collect();
            self.devices[r].bind_thread()?;
            for src in &srcs {
                self.devices[r].add_inplace_f32(&dst, src)?;
            }
        }
        Ok(())
    }
}

fn uniform_split_counts(n_devices: usize, n_layers: usize) -> Vec<usize> {
    let base = n_layers / n_devices;
    let rem = n_layers % n_devices;
    (0..n_devices)
        .map(|i| base + if i < rem { 1 } else { 0 })
        .collect()
}

/// Resolve logical device IDs after the physical `hardware.devices` list has
/// been installed as `ROCR_VISIBLE_DEVICES` and HIP has received the matching
/// post-filter logical IDs. When unset, take the first `n_devices` visible IDs.
fn resolve_device_ids(n_devices: usize) -> HipResult<Vec<i32>> {
    if let Some(ref s) = crate::config::get().devices {
        let ids: Vec<i32> = s
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<i32>())
            .collect::<Result<_, _>>()
            .map_err(|e| HipError::new(0, &format!("hardware.devices parse: {e}")))?;
        if ids.len() < n_devices {
            return Err(HipError::new(
                0,
                &format!(
                    "hardware.devices exposes {} ids but n_devices = {n_devices}",
                    ids.len(),
                ),
            ));
        }
        return Ok(ids[..n_devices].to_vec());
    }
    Ok((0..n_devices as i32).collect())
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
    let allow_mixed = crate::config::get().allow_mixed_arch;
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
    let tol_gb = crate::config::get()
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
}
