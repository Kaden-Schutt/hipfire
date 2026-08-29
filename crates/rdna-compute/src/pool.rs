// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU memory pool — eliminates hipMalloc/hipFree overhead in the hot loop.
//! Pre-allocates buffers of common sizes and reuses them via a free list.

use hip_bridge::{DeviceBuffer, HipResult, HipRuntime};
use std::collections::HashMap;

const MIN_ALLOC: usize = 256;

/// A pool of GPU buffers, bucketed by size.
/// Requesting a buffer returns one from the pool (if available) or allocates new.
/// Returning a buffer puts it back in the pool for reuse.
pub struct GpuPool {
    /// Free buffers bucketed by size (rounded up to power of 2)
    free_lists: HashMap<usize, Vec<DeviceBuffer>>,
    /// Total bytes currently allocated (for diagnostics)
    pub total_allocated: usize,
    pub total_reused: usize,
    pub total_new: usize,
}

impl GpuPool {
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
            total_reused: 0,
            total_new: 0,
        }
    }

    /// Free-list bucket key. Buffers group by power-of-2 bucket so a
    /// decode-hot scratch of size X reliably finds a reusable slot from
    /// a previous step. The bucket is ONLY a reuse key — the actual HIP
    /// allocation uses the exact requested size (see `alloc`), so there
    /// is no VRAM padding waste.
    fn bucket_key(size: usize) -> usize {
        const MIN: usize = 256;
        if size <= MIN {
            MIN
        } else {
            size.next_power_of_two()
        }
    }

    /// Get a buffer of at least `size` bytes. Reuses from the free-list
    /// if a pooled buffer in the same bucket is large enough; otherwise
    /// allocates from HIP at the EXACT requested size.
    ///
    /// Exact HIP allocation matters for large buffers: previously,
    /// target's 15 GB of per-layer weights on 27B sprawled into
    /// ~100–500 MB power-of-2 buckets that each padded up to 2×,
    /// leaving no contiguous room for the ~3.5 GB draft to load on
    /// 24 GB cards. With exact sizing the padding is zero, all
    /// intended bytes are used, and the draft fits.
    pub fn alloc(&mut self, hip: &HipRuntime, size: usize) -> HipResult<DeviceBuffer> {
        let bucket = Self::bucket_key(size);
        if let Some(list) = self.free_lists.get_mut(&bucket) {
            // Pop buffers until we find one with enough capacity. Smaller
            // pooled buffers (from prior smaller requests) are returned
            // to HIP — better to re-allocate at the right size than to
            // carry undersized buffers around.
            while let Some(buf) = list.pop() {
                if buf.size() >= size {
                    self.total_reused += 1;
                    return Ok(buf);
                }
                let _ = hip.free(buf);
            }
        }
        // No suitable buffer — allocate at exact requested size. Round up
        // to the nearest 256 B for alignment; HIP may round further but
        // this keeps our accounting honest and avoids tiny-alloc churn.
        let actual = if size < MIN_ALLOC { MIN_ALLOC } else { size };
        self.total_new += 1;
        self.total_allocated += actual;
        hip.malloc(actual)
    }

    /// Return a buffer to the pool for reuse. The buffer's ACTUAL
    /// capacity is what gets reused — we key the free-list by the
    /// power-of-2 bucket so same-size-shaped requests hit the same
    /// slot.
    pub fn free(&mut self, buf: DeviceBuffer) {
        let bucket = Self::bucket_key(buf.size());
        self.free_lists.entry(bucket).or_default().push(buf);
    }
    /// Return the exact bytes currently held in the reusable free lists.
    ///
    /// This is the cached portion of the pool, not cumulative allocation
    /// traffic (`total_allocated`) and not a logical tensor-size estimate.
    /// Callers use it for developer-only lifecycle accounting before/after
    /// model teardown.
    pub fn cached_bytes(&self) -> usize {
        self.free_lists
            .values()
            .flat_map(|buffers| buffers.iter())
            .map(DeviceBuffer::size)
            .sum()
    }

    /// Actually free all pooled buffers (call on cleanup).
    pub fn drain(&mut self, hip: &HipRuntime) {
        for (_, list) in self.free_lists.drain() {
            for buf in list {
                let _ = hip.free(buf);
            }
        }
    }

    /// Free all pooled buffers using [`HipRuntime::free_preserving`] so that
    /// every buffer whose `hipFree` fails is re-inserted into the pool and
    /// the error is reported rather than lost.  The pool retains ownership
    /// of failed buffers for retry.
    ///
    /// Returns `Ok(())` when all buffers were freed, or an aggregate error
    /// with the count of failures.  The pool is still usable after a partial
    /// failure — the caller can retry [`drain_checked`] or continue using the
    /// pool (failed buffers remain in their free-list bucket).
    pub fn drain_checked(&mut self, hip: &HipRuntime) -> Result<(), String> {
        // Collect all buffers into one flat Vec, then drain free_lists
        // completely before attempting any free. This avoids borrowing
        // self.free_lists mutably while iterating.
        let all_bufs: Vec<DeviceBuffer> =
            self.free_lists.drain().flat_map(|(_, list)| list).collect();
        let mut failures: usize = 0;
        let mut first_err = String::new();
        for buf in all_bufs {
            match hip.free_preserving(buf) {
                Ok(()) => {}
                Err((returned_buf, e)) => {
                    failures += 1;
                    if first_err.is_empty() {
                        first_err = format!("{e:?}");
                    }
                    // Reinsert the failed buffer into the pool so ownership
                    // is preserved and the caller can retry.
                    let bucket = Self::bucket_key(returned_buf.size());
                    self.free_lists
                        .entry(bucket)
                        .or_default()
                        .push(returned_buf);
                }
            }
        }
        if failures == 0 {
            Ok(())
        } else {
            Err(format!(
                "{failures} pool buffer(s) failed to drain (first error: {first_err})"
            ))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::GpuPool;

    #[test]
    fn cached_bytes_reports_actual_free_list_capacity() {
        let mut pool = GpuPool::new();
        pool.free(unsafe {
            hip_bridge::DeviceBuffer::from_raw(0x1000 as *mut std::ffi::c_void, 512)
        });
        pool.free(unsafe {
            hip_bridge::DeviceBuffer::from_raw(0x2000 as *mut std::ffi::c_void, 768)
        });

        assert_eq!(pool.cached_bytes(), 1280);
    }
}
