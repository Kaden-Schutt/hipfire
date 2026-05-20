//! Per-kernel bandwidth profiling.
//!
//! Wraps GPU kernel launches with hipEvent timing + analytical byte counts,
//! recording each launch into a thread-local `Vec<ProfileEntry>`. Enable via
//! `profile::start()`, run the code under test, drain results with
//! `profile::stop()`.
//!
//! When profiling is NOT active, the `begin_timer` / `end_timer` calls are
//! effectively no-ops (one atomic load) — safe to leave in hot paths.
//!
//! The timing approach serializes kernel launches by synchronizing on the
//! stop event after each launch. This measures per-kernel wall time
//! accurately but loses any async pipelining the runtime would have done.
//! For bandwidth attribution this is exactly what we want.

use hip_bridge::{Event, HipResult, HipRuntime};
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub category: &'static str,
    pub kernel: &'static str,
    pub time_us: f64,
    pub bytes: usize,
}

thread_local! {
    static PROFILE: RefCell<Option<Vec<ProfileEntry>>> = const { RefCell::new(None) };
}

/// Start collecting profile entries. Clears any prior state.
pub fn start() {
    PROFILE.with(|p| *p.borrow_mut() = Some(Vec::with_capacity(2048)));
}

/// Stop profiling and return the collected entries. Returns None if profiling
/// was never started (or already stopped).
pub fn stop() -> Option<Vec<ProfileEntry>> {
    PROFILE.with(|p| p.borrow_mut().take())
}

/// Quick check used by launch helpers to skip the event dance when profiling
/// is disabled. This is the only overhead the hot path pays when profiling is
/// off.
#[inline]
pub fn is_active() -> bool {
    PROFILE.with(|p| p.borrow().is_some())
}

/// Record a single profile entry. Only called from `Timer::finish()`.
fn record(entry: ProfileEntry) {
    PROFILE.with(|p| {
        if let Some(ref mut entries) = *p.borrow_mut() {
            entries.push(entry);
        }
    });
}

/// Holds the start/stop events for one kernel launch. Consumed by `finish()`.
pub struct Timer {
    category: &'static str,
    kernel: &'static str,
    bytes: usize,
    start: Event,
    stop: Event,
}

impl Timer {
    /// Finalize the timer: record the stop event, sync, compute elapsed ms,
    /// push into the profile collector, destroy the events.
    pub fn finish(self, hip: &HipRuntime) {
        // event_record/sync/elapsed errors during profiling are swallowed —
        // we never want profiling to crash the forward pass.
        let _ = hip.event_record(&self.stop, None);
        let _ = hip.event_synchronize(&self.stop);
        let ms = hip.event_elapsed_ms(&self.start, &self.stop).unwrap_or(0.0);
        record(ProfileEntry {
            category: self.category,
            kernel: self.kernel,
            time_us: ms as f64 * 1000.0,
            bytes: self.bytes,
        });
        let _ = hip.event_destroy(self.start);
        let _ = hip.event_destroy(self.stop);
    }
}

/// Create start/stop events and record the start event on the null stream.
/// Returns `None` if profiling is not active (hot-path fast case).
///
/// Usage pattern:
/// ```ignore
/// let t = profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256", bytes);
/// unsafe { self.hip.launch_kernel(..., None, ...) }?;
/// if let Some(t) = t { t.finish(&self.hip); }
/// ```
pub fn begin_timer(
    hip: &HipRuntime,
    category: &'static str,
    kernel: &'static str,
    bytes: usize,
) -> Option<Timer> {
    if !is_active() {
        return None;
    }
    let start = hip.event_create().ok()?;
    let stop = hip.event_create().ok()?;
    if hip.event_record(&start, None).is_err() {
        let _ = hip.event_destroy(start);
        let _ = hip.event_destroy(stop);
        return None;
    }
    Some(Timer {
        category,
        kernel,
        bytes,
        start,
        stop,
    })
}

/// Helper that finalizes the timer if present. Convenience wrapper for the
/// common `if let Some(t) = timer { t.finish(&self.hip); }` pattern.
#[inline]
pub fn end_timer(hip: &HipRuntime, timer: Option<Timer>) -> HipResult<()> {
    if let Some(t) = timer {
        t.finish(hip);
    }
    Ok(())
}

// ─── Byte count formulas for common kernel shapes ──────────────────────────
//
// Each helper takes kernel dimensions and returns the number of bytes the
// kernel reads + writes from global memory. These are ANALYTICAL counts —
// we're not measuring DRAM transactions, we're counting the bytes the kernel
// would need in the best case. Real bandwidth utilization = bytes / wall_time.

/// HFQ4-G256 weight matrix: 136 bytes per group of 256 weights.
pub fn hfq4g256_weight_bytes(m: usize, k: usize) -> usize {
    let groups = k / 256;
    m * groups * 136
}

/// Bytes for a single-row GEMV: weight + input vector + output vector.
pub fn gemv_hfq4g256_bytes(m: usize, k: usize) -> usize {
    hfq4g256_weight_bytes(m, k) + k * 4 + m * 4
}

/// MQ3-Lloyd GEMV bytes: weight (112 B / group) + x + y.
pub fn gemv_mq3g256_lloyd_bytes(m: usize, k: usize) -> usize {
    let groups = k / 256;
    m * groups * 112 + k * 4 + m * 4
}

/// Bytes for a B-way batched HFQ4-G256 GEMM (weight read once, B input/output
/// vectors).
pub fn gemm_hfq4g256_bytes(m: usize, k: usize, batch: usize) -> usize {
    hfq4g256_weight_bytes(m, k) + batch * (k + m) * 4
}

/// HFP4-G32 weight footprint: 16-B row header + (K/32)*17-B blocks per row.
pub fn hfp4g32_weight_bytes(m: usize, k: usize) -> usize {
    let blocks = k / 32;
    m * (16 + blocks * 17)
}

/// Single-row HFP4-G32 GEMV bytes: weight + x (FP32) + y (FP32).
pub fn gemv_hfp4g32_bytes(m: usize, k: usize) -> usize {
    hfp4g32_weight_bytes(m, k) + k * 4 + m * 4
}

/// FWHT rotation kernel: read x, read two sign tables, write x_rot.
pub fn mq_rotate_bytes(k: usize) -> usize {
    k * 4 + 256 * 4 * 2 + k * 4
}

/// RMSNorm: read x, read weight, write out.
pub fn rmsnorm_bytes(n: usize) -> usize {
    n * 4 * 3
}

/// Elementwise binary op (add_inplace, silu_mul, mul): 2 reads + 1 write.
pub fn elementwise_bytes(n: usize) -> usize {
    n * 4 * 3
}

/// Single-input elementwise (sigmoid, scale, l2_norm in place): 1 read + 1 write.
pub fn elementwise1_bytes(n: usize) -> usize {
    n * 4 * 2
}

/// DeltaNet Q8 recurrence: roughly state in + state out + Q/K/V + gate/beta +
/// output. Dominated by state read+write.
pub fn gated_delta_net_q8_bytes(
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> usize {
    let state_bytes = n_heads * head_dim * head_dim; // Q8: 1 byte each
    let state_scales = n_heads * head_dim * 4;
    let qkv = 3 * n_tokens * n_heads * head_dim * 4;
    let gate_beta = 2 * n_tokens * n_heads * 4;
    let out = n_tokens * n_heads * head_dim * 4;
    // State is read + written
    2 * state_bytes + 2 * state_scales + qkv + gate_beta + out
}

/// Q8_0 KV attention: read Q, read K+V caches, write output.
/// `kv_len` = current sequence length.
pub fn attention_q8_0_kv_bytes(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> usize {
    let q_bytes = n_heads * head_dim * 4;
    // Q8_0 KV cache = 34 bytes per head_dim=128 block (1 f32 scale + 32 int8).
    // For general head_dim, approximate as head_dim + 4 per head per position.
    let kv_bytes_per_pos = n_kv_heads * (head_dim + 4);
    let kv_bytes = 2 * kv_len * kv_bytes_per_pos;
    let out_bytes = n_heads * head_dim * 4;
    q_bytes + kv_bytes + out_bytes
}

/// RoPE (partial interleaved): read Q, read K, write both back.
pub fn rope_bytes(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> usize {
    (n_heads + n_kv_heads) * head_dim * 4 * 2
}

/// Embedding lookup for HFQ4-G256: reads one row of the embedding table.
pub fn embedding_hfq4g256_bytes(dim: usize) -> usize {
    hfq4g256_weight_bytes(1, dim) + dim * 4
}

/// Conv1D with SiLU, kernel size 4, ring-buffer state: read input + state +
/// weight, write output + state.
pub fn conv1d_silu_bytes(n_channels: usize) -> usize {
    let kernel_size = 4;
    let state_slots = kernel_size - 1;
    n_channels * 4                         // input
        + n_channels * state_slots * 4     // state read
        + n_channels * kernel_size * 4     // weight
        + n_channels * 4                   // output
        + n_channels * state_slots * 4     // state write
}

/// KV cache write (Q8_0 flavor, per token position).
pub fn kv_cache_write_q8_0_bytes(n_kv_heads: usize, head_dim: usize) -> usize {
    // Read source vector + write quantized cache slot.
    let src = n_kv_heads * head_dim * 4;
    let dst = n_kv_heads * (head_dim + 4); // int8 + scale
    src + dst
}

/// Gated norm (L2 + affine): similar bandwidth profile to rmsnorm but with an
/// extra z gate input.
pub fn gated_norm_bytes(n: usize) -> usize {
    // Read x, z, weight. Write out.
    n * 4 * 4
}
