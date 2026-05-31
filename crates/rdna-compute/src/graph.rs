// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Graph-capture lifecycle for AR forward, DFlash verify, and DeltaNet replay.

use hip_bridge::{Graph, GraphExec, HipResult, HipRuntime, Stream};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};

// Thread-local cache of the last device id bound on this thread.
// Shared with `Gpu::bind_thread` / `Gpu::bind_thread_or_warn` in dispatch.rs.
thread_local! {
    pub(crate) static LAST_BOUND_DEVICE: Cell<i32> = const { Cell::new(-1) };
}

/// Bind the given device on the calling thread. Cached via thread_local
/// — only issues `hipSetDevice` when the cached id changes.
#[inline]
pub(crate) fn bind_thread(hip: &HipRuntime, device_id: i32) -> HipResult<()> {
    LAST_BOUND_DEVICE.with(|c| {
        if c.get() != device_id {
            hip.set_device(device_id)?;
            c.set(device_id);
        }
        Ok(())
    })?;
    debug_assert_eq!(
        hip.current_device()?,
        device_id,
        "bind_thread invariant: current device must match device_id",
    );
    Ok(())
}

/// `bind_thread` for infallible / `Drop` contexts. Logs to stderr on
/// `hipSetDevice` failure instead of swallowing it silently.
#[inline]
pub(crate) fn bind_thread_or_warn(hip: &HipRuntime, device_id: i32) {
    LAST_BOUND_DEVICE.with(|c| {
        if c.get() != device_id {
            match hip.set_device(device_id) {
                Ok(()) => c.set(device_id),
                Err(e) => eprintln!(
                    "WARN: bind_thread_or_warn(dev {}) failed: {} — \
                     subsequent ops run on the currently-bound device",
                    device_id, e,
                ),
            }
        }
    });
}

/// Per-B graph cache: verify (DFlash) and replay (DeltaNet tape) share this pattern.
/// Does not implement Clone because `Graph` / `GraphExec` are not Clone.
pub struct PerBGraphCache {
    pub cache: HashMap<usize, (Graph, GraphExec, Vec<Vec<u8>>)>,
    pub warmed_up: HashSet<usize>,
    /// Size being captured right now (between begin_* and end_*). None outside
    /// that window.
    pub capturing: Option<usize>,
    /// Subset of cache entries whose captured region also includes the
    /// DFlash verify lm_head + argmax tail. Callers check this before
    /// deciding whether to enqueue lm_head outside the graph.
    pub lmhead_argmax: HashSet<usize>,
}

/// Graph-capture state split across AR forward, DFlash verify, and DeltaNet replay.
pub struct GraphState {
    // AR forward (single-slot)
    pub capture_mode: bool,
    pub capture_blobs: Vec<Vec<u8>>,
    pub graph_exec: Option<GraphExec>,
    pub captured_graph: Option<Graph>,
    pub ar_forward_kernel_dirty: bool,
    pub ar_forward_replay_enabled: bool,

    // Verify (DFlash, per-B)
    pub verify: PerBGraphCache,

    // Replay (DeltaNet tape, per-n_steps)
    pub replay: PerBGraphCache,
}

impl GraphState {
    // ── hipGraph capture/replay (AR forward) ──────────────────────────────

    /// Begin capturing all kernel launches on the active stream into a graph.
    /// While capturing, dispatch methods that support it will use the blob
    /// launch path so that kernarg pointers survive until graph replay.
    pub fn begin_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        self.capture_blobs.clear();
        self.capture_mode = true;
        hip.stream_begin_capture(stream, 0) // 0 = hipStreamCaptureModeGlobal
    }

    /// End capture, instantiate the graph for replay.
    pub fn end_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        self.capture_mode = false;
        let graph = hip.stream_end_capture(stream)?;
        let exec = hip.graph_instantiate(&graph)?;
        self.captured_graph = Some(graph);
        self.graph_exec = Some(exec);
        Ok(())
    }

    /// Replay the captured graph.
    pub fn graph_launch(
        &self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let exec = self.graph_exec.as_ref().expect("no captured graph to replay");
        hip.graph_launch(exec, stream)
    }

    /// Caller signals end of a decode turn (EOS or max_tokens reached). If a
    /// captured graph exists and kernels are clean, replay is enabled for the
    /// next decode turn. Per the AR-forward hipGraph policy: "at least one
    /// captured full turn must run before replay can be enabled."
    /// No-op if no capture exists (e.g., turn ran fully direct because kernels
    /// were dirty or graph was disabled by the caller).
    pub fn end_decode_turn(&mut self) {
        if !self.ar_forward_kernel_dirty && self.graph_exec.is_some() {
            self.ar_forward_replay_enabled = true;
        }
    }

    /// Drop the currently captured graph (if any) without touching kernel /
    /// replay state. Used by the capture+launch hot-path to free the previous
    /// per-call capture before recording a fresh one — bare `graph_destroy()`
    /// would also mark kernels dirty + disable replay, which is wrong here.
    pub fn drop_captured_graph(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        if let Some(exec) = self.graph_exec.take() {
            let _ = hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.captured_graph.take() {
            let _ = hip.graph_destroy(graph);
        }
        self.capture_blobs.clear();
    }

    /// Caller signals a kernel-module change (model load, dtype switch, etc).
    /// Forces the next AR forward call to dispatch direct (no capture) so any
    /// inline JIT / lazy hipMalloc happens outside a captured region. Replay
    /// stays disabled until a fresh full turn completes via `end_decode_turn`.
    pub fn mark_kernels_dirty(&mut self) {
        self.ar_forward_kernel_dirty = true;
        self.ar_forward_replay_enabled = false;
    }

    /// Destroy the captured graph and free all retained kernarg blobs.
    pub fn graph_destroy(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        if let Some(exec) = self.graph_exec.take() {
            let _ = hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.captured_graph.take() {
            let _ = hip.graph_destroy(graph);
        }
        self.capture_blobs.clear();
        self.ar_forward_kernel_dirty = true;
        self.ar_forward_replay_enabled = false;
    }

    // ── Per-B verify-forward graph cache ─────────────────────────────────

    /// Does a captured verify graph exist for batch size `b`?
    pub fn verify_has_graph(&self, b: usize) -> bool {
        self.verify.cache.contains_key(&b)
    }

    /// Does `b` need a warmup pass before capture can begin?
    pub fn verify_needs_warmup(&self, b: usize) -> bool {
        !self.verify.warmed_up.contains(&b)
    }

    /// Mark `b` as having completed its warmup.
    pub fn verify_mark_warmup_done(&mut self, b: usize) {
        self.verify.warmed_up.insert(b);
    }

    /// Begin capturing a verify-forward graph for batch size `b`. Subsequent
    /// `launch_maybe_blob` calls will push their kernargs into `capture_blobs`,
    /// which is drained into the per-B cache entry on `end_verify_graph_capture`.
    pub fn begin_verify_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        b: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        debug_assert!(self.verify.capturing.is_none(),
            "begin_verify_graph_capture: already capturing for b={:?}",
            self.verify.capturing);
        debug_assert!(!self.capture_mode,
            "begin_verify_graph_capture: capture_mode already set");
        self.capture_blobs.clear();
        self.verify.capturing = Some(b);
        self.capture_mode = true;
        hip.stream_begin_capture(stream, 0) // hipStreamCaptureModeGlobal
    }

    /// End capture, instantiate, stash into the per-B cache (taking ownership
    /// of the current `capture_blobs`).
    pub fn end_verify_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let b = self.verify.capturing.take()
            .expect("end_verify_graph_capture without matching begin");
        self.capture_mode = false;
        let graph = hip.stream_end_capture(stream)?;
        let exec = hip.graph_instantiate(&graph)?;
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.verify.cache.insert(b, (graph, exec, blobs));
        Ok(())
    }

    /// Replay the cached verify graph for batch size `b`.
    pub fn verify_graph_launch(
        &self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        b: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let entry = self.verify.cache.get(&b)
            .unwrap_or_else(|| panic!("no captured verify graph for b={}", b));
        hip.graph_launch(&entry.1, stream)
    }

    /// How many captured verify graphs are in the cache (for debug logs).
    pub fn verify_graph_count(&self) -> usize {
        self.verify.cache.len()
    }

    /// Does the captured verify graph for `b` include the lm_head + argmax tail?
    pub fn verify_graph_has_lmhead_argmax(&self, b: usize) -> bool {
        self.verify.lmhead_argmax.contains(&b)
    }

    /// Mark the captured verify graph for `b` as including lm_head + argmax.
    pub fn verify_mark_graph_lmhead_argmax(&mut self, b: usize) {
        self.verify.lmhead_argmax.insert(b);
    }

    /// Destroy all cached verify graphs and their blobs.
    pub fn verify_graph_destroy_all(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        for (_, (graph, exec, _blobs)) in self.verify.cache.drain() {
            let _ = hip.graph_exec_destroy(exec);
            let _ = hip.graph_destroy(graph);
        }
        self.verify.warmed_up.clear();
        self.verify.lmhead_argmax.clear();
        self.verify.capturing = None;
    }

    // ── Replay-graph cache (tape replay after verify) ────────────────────

    /// Does a captured replay graph exist for `n_steps`?
    pub fn replay_has_graph(&self, n_steps: usize) -> bool {
        self.replay.cache.contains_key(&n_steps)
    }

    /// Does `n_steps` need a warmup pass before capture can begin?
    pub fn replay_needs_warmup(&self, n_steps: usize) -> bool {
        !self.replay.warmed_up.contains(&n_steps)
    }

    /// Mark `n_steps` as having completed its warmup.
    pub fn replay_mark_warmup_done(&mut self, n_steps: usize) {
        self.replay.warmed_up.insert(n_steps);
    }

    /// Begin capturing a replay graph for `n_steps`.
    pub fn begin_replay_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        n_steps: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        debug_assert!(self.replay.capturing.is_none(),
            "begin_replay_graph_capture: already capturing for n_steps={:?}",
            self.replay.capturing);
        debug_assert!(!self.capture_mode,
            "begin_replay_graph_capture: capture_mode already set");
        self.capture_blobs.clear();
        self.replay.capturing = Some(n_steps);
        self.capture_mode = true;
        hip.stream_begin_capture(stream, 0)
    }

    /// End capture, instantiate, stash into the per-n_steps cache.
    pub fn end_replay_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let n_steps = self.replay.capturing.take()
            .expect("end_replay_graph_capture without matching begin");
        self.capture_mode = false;
        let graph = hip.stream_end_capture(stream)?;
        let exec = hip.graph_instantiate(&graph)?;
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.replay.cache.insert(n_steps, (graph, exec, blobs));
        Ok(())
    }

    /// Replay the cached replay graph for `n_steps`.
    pub fn replay_graph_launch(
        &self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        n_steps: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let entry = self.replay.cache.get(&n_steps)
            .unwrap_or_else(|| panic!("no captured replay graph for n_steps={}", n_steps));
        hip.graph_launch(&entry.1, stream)
    }

    /// How many captured replay graphs are in the cache (for debug logs).
    pub fn replay_graph_count(&self) -> usize {
        self.replay.cache.len()
    }

    /// Destroy all cached replay graphs and their blobs.
    pub fn replay_graph_destroy_all(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        for (_, (graph, exec, _blobs)) in self.replay.cache.drain() {
            let _ = hip.graph_exec_destroy(exec);
            let _ = hip.graph_destroy(graph);
        }
        self.replay.warmed_up.clear();
        self.replay.capturing = None;
    }
}
