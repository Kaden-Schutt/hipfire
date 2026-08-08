// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic checked-free teardown machinery shared by every arch bundle.
//!
//! The loader's unload and rollback paths must never drop a still-allocated
//! GPU owner: `GpuTensor`/`DeviceBuffer` have no `Drop`. These types make
//! teardown *owner-preserving* at the type level — every free is attempted
//! through `Gpu::free_tensor_checked` and every owner that could not be
//! freed is carried in a [`GpuCleanupFailure`] for exact-retention retry.
//!
//! - [`RetainedGpuTensor`]: a `GpuTensor` whose checked free failed.
//! - [`RetryableOwner`]: category-preserving retry for NON-tensor GPU
//!   owners (e.g. a frozen weight store) — the generic failure container
//!   holds them as trait objects so a bundle can carry mixed owner kinds.
//! - [`GpuCleanupFailure`]: the aggregate returned by every checked teardown.
//! - [`BundleTeardown`]: the per-bundle trait the loader dispatches on.
//! - [`free_tensor_retained`] / [`free_weight_all_checked`] /
//!   [`retain_free!`]: the shared checked-free helpers.

use crate::llama::WeightTensor;
use crate::weight_store::SingleFreeFailed;
use rdna_compute::{Gpu, GpuTensor};

/// A GPU tensor whose checked free failed.
///
/// Retains the original `GpuTensor` — the caller can inspect it or retry.
/// Constructed only by the checked-free family; never by hand.
pub struct RetainedGpuTensor {
    pub label: String,
    pub tensor: GpuTensor,
    pub last_error: String,
}

impl RetainedGpuTensor {
    /// Descriptive label identifying which field this tensor belongs to.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The original GPU tensor that could not be freed.
    pub fn tensor(&self) -> &GpuTensor {
        &self.tensor
    }

    /// Human-readable error from the failed free attempt.
    pub fn last_error(&self) -> &str {
        &self.last_error
    }

    /// Retry freeing this tensor. On success the tensor is consumed.
    /// On failure the tensor is returned alongside the new error.
    pub fn retry(mut self, gpu: &mut Gpu) -> Result<(), RetainedGpuTensor> {
        let mut opt = Some(self.tensor);
        match gpu.free_tensor_checked(&mut opt) {
            Ok(()) => {
                // Tensor was taken by free_tensor_checked on success.
                Ok(())
            }
            Err(e) => {
                self.last_error = e.to_string();
                self.tensor = opt
                    .take()
                    .expect("free_tensor_checked failed but left Option empty — this is a bug");
                Err(self)
            }
        }
    }
}

impl std::fmt::Debug for RetainedGpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedGpuTensor")
            .field("label", &self.label)
            .field("last_error", &self.last_error)
            .finish()
    }
}

/// A non-tensor GPU owner whose checked free failed (e.g. a frozen weight
/// store). Category-preserving retry: the concrete owner type decides how
/// its own cleanup is retried, so the generic [`GpuCleanupFailure`] can
/// carry mixed owner kinds without flattening them.
pub trait RetryableOwner: std::fmt::Debug + Send {
    /// Retry freeing this owner. On success it is consumed; on failure the
    /// owner is returned (same category, never flattened).
    fn retry_boxed(self: Box<Self>, gpu: &mut Gpu) -> Result<(), Box<dyn RetryableOwner>>;

    /// Number of allocations that still need to be freed.
    fn num_failed(&self) -> usize;

    /// Human-readable diagnostic summaries for every failure.
    fn error_summaries(&self) -> Vec<String>;
}

/// A frozen single-store owner (failed `SingleFrozenWeightStore` free) is
/// a category of its own: it holds frozen weight owners, not plain
/// tensors, and must never be flattened into [`RetainedGpuTensor`]s.
impl RetryableOwner for SingleFreeFailed {
    fn retry_boxed(self: Box<Self>, gpu: &mut Gpu) -> Result<(), Box<dyn RetryableOwner>> {
        match self.retry(gpu) {
            Ok(()) => Ok(()),
            Err(f) => Err(Box::new(f)),
        }
    }

    fn num_failed(&self) -> usize {
        self.num_failed()
    }

    fn error_summaries(&self) -> Vec<String> {
        self.error_summaries()
    }
}

/// Aggregate of all cleanup failures from a checked teardown.
///
/// Contains individual failed tensors as [`RetainedGpuTensor`] entries plus
/// any non-tensor owners (via [`RetryableOwner`]). Successful frees are
/// consumed and never appear here.
pub struct GpuCleanupFailure {
    pub failed_tensors: Vec<RetainedGpuTensor>,
    pub other: Vec<Box<dyn RetryableOwner>>,
}

impl GpuCleanupFailure {
    /// Create an empty failure (no failed allocations).
    pub fn empty() -> Self {
        Self {
            failed_tensors: Vec::new(),
            other: Vec::new(),
        }
    }

    /// True when no allocations failed.
    pub fn is_empty(&self) -> bool {
        self.failed_tensors.is_empty() && self.other.is_empty()
    }

    /// Add a single retained tensor to this failure.
    pub fn add_retained(&mut self, retained: RetainedGpuTensor) {
        self.failed_tensors.push(retained);
    }

    /// Add a single non-tensor owner to this failure.
    pub fn add_other(&mut self, owner: Box<dyn RetryableOwner>) {
        self.other.push(owner);
    }

    /// Total number of failed allocations.
    pub fn num_failed(&self) -> usize {
        self.failed_tensors.len() + self.other.iter().map(|o| o.num_failed()).sum::<usize>()
    }

    /// Human-readable diagnostic summaries for every failure.
    pub fn error_summaries(&self) -> Vec<String> {
        let mut summaries: Vec<String> = self
            .failed_tensors
            .iter()
            .map(|r| format!("{}: {}", r.label, r.last_error))
            .collect();
        for o in &self.other {
            summaries.extend(o.error_summaries());
        }
        summaries
    }

    /// Merge another [`GpuCleanupFailure`] into this one.
    ///
    /// Every failed item from `other` is appended — no first-wins/drop
    /// semantics. If both sides carry non-tensor owners, both are retained
    /// for independent retry.
    pub fn merge(&mut self, other: GpuCleanupFailure) {
        self.failed_tensors.extend(other.failed_tensors);
        self.other.extend(other.other);
    }

    /// Retry every retained allocation. Continues after failures.
    ///
    /// On success all resources are consumed. On failure the remaining
    /// failures are returned in a new [`GpuCleanupFailure`] — any
    /// successful retries are consumed and must not be retried again.
    pub fn retry(mut self, gpu: &mut Gpu) -> Result<(), GpuCleanupFailure> {
        let mut failures = Vec::new();
        for r in self.failed_tensors {
            match r.retry(gpu) {
                Ok(()) => {} // consumed
                Err(r) => failures.push(r),
            }
        }
        self.failed_tensors = failures;

        // Retry every non-tensor owner, keep only those that fail again.
        let mut other_failures = Vec::new();
        for o in self.other {
            match o.retry_boxed(gpu) {
                Ok(()) => {} // consumed
                Err(o) => other_failures.push(o),
            }
        }
        self.other = other_failures;

        if self.failed_tensors.is_empty() && self.other.is_empty() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Debug for GpuCleanupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redacted: counts + summaries only. Never stringify the retained
        // owners (they must be retried, never dropped after a log).
        f.debug_struct("GpuCleanupFailure")
            .field("num_failed", &self.num_failed())
            .field("summaries", &self.error_summaries())
            .finish()
    }
}

// ── Continue-and-retain cleanup helpers ─────────────────────────────

/// Production-used generic helper: given a (label, GpuTensor) pair, move
/// the tensor into an `Option` and call `free_tensor_checked`, collecting
/// the [`RetainedGpuTensor`] on failure.
///
/// # GPU evidence limitation
///
/// CPU tests using `GpuTensor::null_for_test()` exercise the ownership
/// retention logic (label, identity, retry) but cannot prove that the HIP
/// `hipFree` call actually succeeds or fails — that requires a real GPU
/// with controlled error injection.
///
/// # Safety
///
/// The tensor is always taken from the `Option` by `free_tensor_checked`,
/// so on success it is no longer accessible. On failure the original
/// `GpuTensor` is preserved in [`RetainedGpuTensor`].
pub fn free_tensor_retained(
    label: impl Into<String>,
    tensor: GpuTensor,
    gpu: &mut Gpu,
    failures: &mut Vec<RetainedGpuTensor>,
) {
    let label = label.into();
    let mut opt = Some(tensor);
    if let Err(e) = gpu.free_tensor_checked(&mut opt) {
        // free_tensor_checked only returns Err when bind_thread fails,
        // which happens BEFORE the tensor is taken from the Option.
        // If the Option is None here, it's an invariant violation in
        // the GPU driver/checked-free contract — panic with a precise
        // message matching RetainedGpuTensor::retry.
        let t = opt.take().expect(
            "free_tensor_retained: free_tensor_checked returned Err but consumed the tensor — this is a bug",
        );
        failures.push(RetainedGpuTensor {
            label,
            tensor: t,
            last_error: e.to_string(),
        });
    }
}

/// Continue-and-retain helper for `WeightTensor`: free all owned buffers
/// (buf, paro sidecars, AWQ scale). Skips aliased Paro rotations.
pub fn free_weight_all_checked(
    label: &str,
    wt: WeightTensor,
    gpu: &mut Gpu,
    failures: &mut Vec<RetainedGpuTensor>,
) {
    // Paro sidecars (skip aliased — the shared owner frees them).
    if let Some(paro) = wt.paro {
        if !paro.is_alias {
            free_tensor_retained(format!("{label}.paro.pairs"), paro.pairs, gpu, failures);
            free_tensor_retained(format!("{label}.paro.theta"), paro.theta, gpu, failures);
            free_tensor_retained(
                format!("{label}.paro.channel_scales"),
                paro.channel_scales,
                gpu,
                failures,
            );
        }
    }
    // AWQ sidecar.
    if let Some(awq) = wt.awq_scale {
        free_tensor_retained(format!("{label}.awq_scale"), awq, gpu, failures);
    }
    free_tensor_retained(format!("{label}.buf"), wt.buf, gpu, failures);
}

/// Continue-and-retain helper for `WeightTensor`: free only sidecars
/// (paro, AWQ).  Used when the main buffer is a non-owning alias
/// (tied lm_head).
pub fn free_weight_sidecars_checked(
    label: &str,
    wt: WeightTensor,
    gpu: &mut Gpu,
    failures: &mut Vec<RetainedGpuTensor>,
) {
    if let Some(paro) = wt.paro {
        if !paro.is_alias {
            free_tensor_retained(format!("{label}.paro.pairs"), paro.pairs, gpu, failures);
            free_tensor_retained(format!("{label}.paro.theta"), paro.theta, gpu, failures);
            free_tensor_retained(
                format!("{label}.paro.channel_scales"),
                paro.channel_scales,
                gpu,
                failures,
            );
        }
    }
    if let Some(awq) = wt.awq_scale {
        free_tensor_retained(format!("{label}.awq_scale"), awq, gpu, failures);
    }
}

/// Fold the failures of a [`crate::llama::KvCache::free_checked`] call into
/// a retained-tensor list. Success (or a cache with nothing left to free)
/// is a no-op.
pub fn retain_kv_failures(
    kv_result: Result<(), Vec<(String, GpuTensor)>>,
    failures: &mut Vec<RetainedGpuTensor>,
) {
    if let Err(remaining) = kv_result {
        for (label, tensor) in remaining {
            failures.push(RetainedGpuTensor {
                label,
                tensor,
                last_error: "kv free_checked failed".into(),
            });
        }
    }
}

/// Declarative checked teardown for GPU-owning structs: every
/// `label => expr` pair is an `Option<GpuTensor>` (write `Some(field)` for
/// non-optional tensors) whose checked free is attempted, retaining failures.
///
/// ```ignore
/// retain_free!(gpu, failures,
///     "MyState.x" => Some(self.x),
///     "MyState.opt_y" => self.opt_y,
/// );
/// ```
#[macro_export]
macro_rules! retain_free {
    ($gpu:expr, $failures:expr, $($label:expr => $tensor:expr),* $(,)?) => {
        $(
            if let Some(t) = $tensor {
                $crate::gpu_cleanup::free_tensor_retained($label, t, $gpu, &mut $failures);
            }
        )*
    };
}

/// Generic checked teardown for arch bundles.
///
/// Every GPU owner is freed with CHECKED frees; on success all resources are
/// consumed (`Ok(())`). On failure the returned [`GpuCleanupFailure`]
/// carries every owner that could not be freed for exact-retention retry —
/// no best-effort `let _ =` free as a correctness mechanism.
pub trait BundleTeardown {
    fn free_checked(self, gpu: &mut Gpu) -> Result<(), GpuCleanupFailure>;
}

// ── Process-local retained-owner backlog ──────────────────────────────

/// The loader's String-returning surfaces (`load_model`, `unload_model`,
/// carrier error paths) cannot carry [`GpuCleanupFailure`] owners onward.
/// Instead of dropping them while allocated, the terminal paths ENQUEUE them
/// here — the backlog is the exact-retention owner until a retry point
/// drains it. [`retry_backlog`] runs at every load/unload boundary, so a
/// later operation reclaims whatever an earlier one could not free.
///
/// The backlog holds whole [`GpuCleanupFailure`] entries (both categories —
/// tensors and boxed [`RetryableOwner`]s travel together, never flattened).
use std::sync::Mutex;

static RETAINED_BACKLOG: Mutex<Vec<GpuCleanupFailure>> = Mutex::new(Vec::new());

/// Enqueue a whole owner-carrying cleanup failure that a String-returning
/// API cannot carry onward. Owners are retained here (never dropped while
/// allocated) until the next retry point.
pub fn enqueue_cleanup_failure(cf: GpuCleanupFailure) {
    RETAINED_BACKLOG
        .lock()
        .expect("retained-owner backlog poisoned")
        .push(cf);
}

/// Enqueue a single retained tensor (e.g. a per-owner retry failure at a
/// String-returning boundary).
pub fn enqueue_retained(r: RetainedGpuTensor) {
    let mut cf = GpuCleanupFailure::empty();
    cf.add_retained(r);
    enqueue_cleanup_failure(cf);
}

/// Retry every enqueued owner once. Successful retries are consumed; owners
/// that still fail are re-enqueued. Returns the number of allocations still
/// pending (0 = backlog empty).
pub fn retry_backlog(gpu: &mut Gpu) -> usize {
    let mut backlog = RETAINED_BACKLOG
        .lock()
        .expect("retained-owner backlog poisoned");
    let mut still = Vec::new();
    for cf in backlog.drain(..) {
        match cf.retry(gpu) {
            Ok(()) => {} // consumed
            Err(remaining) => still.push(remaining),
        }
    }
    let pending: usize = still.iter().map(|cf| cf.num_failed()).sum();
    *backlog = still;
    pending
}

/// Number of allocations currently pending in the backlog (diagnostics and
/// tests).
pub fn backlog_pending() -> usize {
    RETAINED_BACKLOG
        .lock()
        .expect("retained-owner backlog poisoned")
        .iter()
        .map(|cf| cf.num_failed())
        .sum()
}
