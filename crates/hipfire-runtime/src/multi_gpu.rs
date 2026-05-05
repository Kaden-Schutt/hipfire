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
    DeviceBuffer, Event, HipError, HipResult,
    HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
use rdna_compute::{Gpu, GpuTensor};

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
        preflight_vram(&devices)?;
        let per_device = uniform_split_counts(n_devices, n_layers);
        Self::from_parts(devices, per_device, n_layers)
    }

    /// Explicit escape hatch for asymmetric VRAM / hand-tuned splits. Same
    /// pre-flight checks as `init_uniform`. `per_device` length determines
    /// `n_devices`; sum determines `n_layers`.
    pub fn init_layers(per_device: &[usize]) -> HipResult<Self> {
        let n_devices = per_device.len();
        if n_devices == 0 {
            return Err(HipError::new(0, "init_layers: per_device must be non-empty"));
        }
        if per_device.contains(&0) {
            return Err(HipError::new(0, "init_layers: each device must own ≥1 layer"));
        }
        let n_layers: usize = per_device.iter().sum();
        let device_ids = resolve_device_ids(n_devices)?;
        let devices = construct_devices(&device_ids)?;
        preflight_vram(&devices)?;
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
            devices: vec![gpu],
            layer_to_device: vec![0; n_layers],
            band_starts: vec![0],
            peer_access_enabled: false,
            output_device: 0,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
        }
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
                match self.devices[i].hip.enable_peer_access(self.devices[j].device_id) {
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
                src_gpu.hip.memcpy_peer_async(
                    dst, dst_dev_id, src, src_dev_id, n_bytes, stream,
                )?;
                let event = src_gpu.hip.event_create()?;
                match src_gpu.hip.event_record(&event, Some(stream)) {
                    Ok(()) => Ok(BoundaryEvent { dst_dev, completion: Some(event) }),
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
                src_gpu.hip.memcpy_peer(dst, dst_dev_id, src, src_dev_id, n_bytes)?;
                Ok(BoundaryEvent { dst_dev, completion: None })
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

    fn from_parts(
        devices: Vec<Gpu>,
        per_device: Vec<usize>,
        n_layers: usize,
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
            devices,
            layer_to_device,
            band_starts,
            peer_access_enabled: false,
            output_device: n_devices - 1,
            givens_cos_per_dev: Vec::new(),
            givens_sin_per_dev: Vec::new(),
        })
    }
}

fn uniform_split_counts(n_devices: usize, n_layers: usize) -> Vec<usize> {
    let base = n_layers / n_devices;
    let rem = n_layers % n_devices;
    (0..n_devices)
        .map(|i| base + if i < rem { 1 } else { 0 })
        .collect()
}

/// Resolve the device IDs to use. Logical IDs post-`HIP_VISIBLE_DEVICES`:
/// `HIPFIRE_DEVICES=0,1` selects the first two HIP-visible devices. When
/// unset, takes the first `n_devices` visible IDs.
fn resolve_device_ids(n_devices: usize) -> HipResult<Vec<i32>> {
    if let Ok(s) = std::env::var("HIPFIRE_DEVICES") {
        let ids: Vec<i32> = s
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<i32>())
            .collect::<Result<_, _>>()
            .map_err(|e| HipError::new(0, &format!("HIPFIRE_DEVICES parse: {e}")))?;
        if ids.len() < n_devices {
            return Err(HipError::new(
                0,
                &format!(
                    "HIPFIRE_DEVICES has {} ids but n_devices = {n_devices}",
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

fn preflight_vram(devices: &[Gpu]) -> HipResult<()> {
    if devices.is_empty() {
        return Ok(());
    }
    let arch0 = devices[0].arch.clone();
    let mut frees = Vec::with_capacity(devices.len());
    for d in devices {
        if d.arch != arch0 {
            return Err(HipError::new(
                0,
                &format!(
                    "preflight_vram: arch mismatch — dev 0 is {arch0}, dev {} is {}. \
                     Mixed-arch is not supported in v1.",
                    d.device_id, d.arch,
                ),
            ));
        }
        d.bind_thread()?;
        let (free, _total) = d.hip.get_vram_info()?;
        frees.push(free);
    }
    let max_free = *frees.iter().max().unwrap();
    let min_free = *frees.iter().min().unwrap();
    let delta_gb = (max_free - min_free) as f64 / 1e9;
    let tol_gb = std::env::var("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
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
