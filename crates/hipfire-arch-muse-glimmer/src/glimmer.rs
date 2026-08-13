// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Weights + state: loading, KV caches, layer mapping.
//!
//! Tensors live under `model.language_model.*` except `lm_head.weight`
//! which is a separate (untied) tensor, NOT an alias of embed_tokens
//! (see `lib.rs`).

use crate::config::GlimmerConfig;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

// ──────────────────── Device hidden capture log ────────────────────

/// Kind of an open device-capture transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenStageKind {
    Prefill,
    Verify,
    Decode,
}

/// Metadata for one open (or verify-ready) capture transaction.
///
/// Staging layout is contiguous and never wraps:
/// `staging[stage_row][ci][d]` for `stage_row ∈ [0, stored_rows)`.
/// Source row `r` maps to stage row `r - stored_row0` when retained.
#[derive(Debug, Clone)]
pub struct HiddenStage {
    pub kind: HiddenStageKind,
    pub start_abs: usize,
    pub total_rows: usize,
    /// First source-row index retained in staging (0 for verify/decode;
    /// `total_rows - stored_rows` for prefill when `total > capacity`).
    pub stored_row0: usize,
    pub stored_rows: usize,
    /// Bit `i` set ⇔ extract slot `i` has been scattered successfully.
    pub captured_mask: u64,
}

/// Transaction state machine for [`GlimmerHiddenLog`].
#[derive(Debug, Clone)]
pub enum HiddenStageState {
    Idle,
    Writing(HiddenStage),
    /// Verify finished scattering every extract slot; committed ring
    /// interval is still unchanged. Awaiting [`GlimmerHiddenLog::commit_verified_prefix`].
    VerifyReady(HiddenStage),
}

/// Device-resident target-hidden ring for Muse Glimmer DFlash capture.
///
/// Physical layout of `rows`: `[capacity_rows][num_extract][hidden]` F32,
/// absolute position `p` → slot `p % capacity_rows`. Staging is a contiguous
/// non-wrapping buffer of `stage_capacity_rows` rows with the same per-row
/// layout. Logical readable interval is always the contiguous half-open
/// range `[valid_abs_start, committed_abs_end)` of length ≤ `capacity_rows`.
///
/// Extract slot `ci` is the caller-supplied capture-layer ordinal (strictly
/// increasing `target_layer_ids` order). It is never re-sorted.
pub struct GlimmerHiddenLog {
    rows: GpuTensor,
    staging: GpuTensor,
    capacity_rows: usize,
    stage_capacity_rows: usize,
    num_extract: usize,
    hidden: usize,
    capture_layers: Vec<usize>,
    /// `layer_to_slot[layer] = Some(ci)` for each capture layer.
    layer_to_slot: Vec<Option<usize>>,
    valid_abs_start: usize,
    committed_abs_end: usize,
    stage: HiddenStageState,
    /// Set when a multi-segment stage→ring commit enqueues at least one D2D
    /// then fails. Logical watermarks are untrustworthy until [`reset`].
    poisoned: bool,
    /// Audit-only host position-major shadow of the open verify transaction
    /// (`[total_rows][num_extract][hidden]`). Populated under
    /// `HIPFIRE_GLIMMER_DEVICE_CAPTURE_AUDIT=1`; cleared on commit/abort/reset.
    audit_host: Option<Vec<f32>>,
}

impl std::fmt::Debug for GlimmerHiddenLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlimmerHiddenLog")
            .field("capacity_rows", &self.capacity_rows)
            .field("stage_capacity_rows", &self.stage_capacity_rows)
            .field("num_extract", &self.num_extract)
            .field("hidden", &self.hidden)
            .field("capture_layers", &self.capture_layers)
            .field("valid_abs_start", &self.valid_abs_start)
            .field("committed_abs_end", &self.committed_abs_end)
            .field("stage", &self.stage)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

// ── pure layout / retention helpers (unit-tested without GPU) ──

/// Prefill retains only the newest `min(total, capacity)` source rows.
#[inline]
pub fn hidden_prefill_retention(total_rows: usize, capacity_rows: usize) -> (usize, usize) {
    let stored_rows = total_rows.min(capacity_rows);
    let stored_row0 = total_rows - stored_rows;
    (stored_row0, stored_rows)
}

/// Verify commit retains the accepted prefix; if `keep > capacity`, keep the
/// newest `capacity` rows of that prefix (source row0 = keep - capacity).
#[inline]
pub fn hidden_verify_retention(keep_rows: usize, capacity_rows: usize) -> (usize, usize) {
    let stored_rows = keep_rows.min(capacity_rows);
    let src_row0 = keep_rows - stored_rows;
    (src_row0, stored_rows)
}

/// Flat F32 index of `(abs_pos, ci, d)` inside a capacity-ring of the given shape.
#[inline]
pub fn hidden_ring_elem_offset(
    abs_pos: usize,
    ci: usize,
    d: usize,
    capacity_rows: usize,
    num_extract: usize,
    hidden: usize,
) -> usize {
    debug_assert!(capacity_rows > 0);
    debug_assert!(ci < num_extract);
    debug_assert!(d < hidden);
    let slot = abs_pos % capacity_rows;
    (slot * num_extract + ci) * hidden + d
}

/// Flat F32 index of `(stage_row, ci, d)` inside non-wrapping staging.
#[inline]
pub fn hidden_stage_elem_offset(
    stage_row: usize,
    ci: usize,
    d: usize,
    num_extract: usize,
    hidden: usize,
) -> usize {
    debug_assert!(ci < num_extract);
    debug_assert!(d < hidden);
    (stage_row * num_extract + ci) * hidden + d
}

/// Whether a rewind of the committed end to `end` is supported given the
/// current readable window. Matches [`GlimmerHiddenLog::can_rewind_to`].
#[inline]
pub fn hidden_can_rewind_to(
    end: usize,
    committed_abs_end: usize,
    valid_abs_start: usize,
    capacity_rows: usize,
) -> bool {
    end <= committed_abs_end && end.saturating_sub(capacity_rows) >= valid_abs_start
}

/// `HIPFIRE_GLIMMER_DEVICE_CAPTURE_AUDIT=1` enables diagnostic host dual-collect
/// and bit-exact staging/ring checks in DEVICE mode. Completely off the hot path
/// unless set; may synchronize and download.
#[inline]
pub fn device_capture_audit_enabled() -> bool {
    std::env::var("HIPFIRE_GLIMMER_DEVICE_CAPTURE_AUDIT")
        .ok()
        .as_deref()
        == Some("1")
}

/// Format a single F32 bit-mismatch for device-capture audit diagnostics.
#[inline]
pub fn audit_bits_mismatch(
    phase: &str,
    abs_row: usize,
    ci: usize,
    layer: usize,
    dim: usize,
    expected_bits: u32,
    actual_bits: u32,
) -> String {
    format!(
        "glimmer device-capture audit mismatch: phase={phase} abs_row={abs_row} \
         capture_slot={ci} layer={layer} dim={dim} expected_bits={expected_bits:#010x} \
         actual_bits={actual_bits:#010x}"
    )
}

/// Compare host position-major capture `[n_rows][ne][hidden]` against a flat
/// device download of the same layout. Pure helper for unit tests and audit.
pub fn audit_compare_position_major(
    phase: &str,
    host: &[f32],
    device: &[f32],
    n_rows: usize,
    num_extract: usize,
    hidden: usize,
    abs_row0: usize,
    capture_layers: &[usize],
) -> Result<(), String> {
    let need = n_rows
        .checked_mul(num_extract)
        .and_then(|v| v.checked_mul(hidden))
        .ok_or_else(|| "glimmer audit: size overflow".to_string())?;
    if host.len() < need || device.len() < need {
        return Err(format!(
            "glimmer device-capture audit {phase}: host len {} device len {} need {need}",
            host.len(),
            device.len()
        ));
    }
    for row in 0..n_rows {
        for ci in 0..num_extract {
            let layer = capture_layers.get(ci).copied().unwrap_or(usize::MAX);
            for d in 0..hidden {
                let off = (row * num_extract + ci) * hidden + d;
                let e = host[off];
                let a = device[off];
                if e.to_bits() != a.to_bits() {
                    return Err(audit_bits_mismatch(
                        phase,
                        abs_row0 + row,
                        ci,
                        layer,
                        d,
                        e.to_bits(),
                        a.to_bits(),
                    ));
                }
            }
        }
    }
    Ok(())
}

impl GlimmerHiddenLog {
    #[inline]
    pub fn capacity_rows(&self) -> usize {
        self.capacity_rows
    }

    #[inline]
    pub fn stage_capacity_rows(&self) -> usize {
        self.stage_capacity_rows
    }

    #[inline]
    pub fn num_extract(&self) -> usize {
        self.num_extract
    }

    #[inline]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    #[inline]
    pub fn capture_layers(&self) -> &[usize] {
        &self.capture_layers
    }

    #[inline]
    pub fn layer_to_slot(&self) -> &[Option<usize>] {
        &self.layer_to_slot
    }

    #[inline]
    pub fn valid_abs_start(&self) -> usize {
        self.valid_abs_start
    }

    #[inline]
    pub fn committed_abs_end(&self) -> usize {
        self.committed_abs_end
    }

    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[inline]
    pub fn stage_is_idle(&self) -> bool {
        matches!(self.stage, HiddenStageState::Idle)
    }

    #[inline]
    pub fn stage(&self) -> &HiddenStageState {
        &self.stage
    }

    /// F32 elements per logical position row: `num_extract * hidden`.
    #[inline]
    pub fn row_elems(&self) -> usize {
        self.num_extract * self.hidden
    }

    #[inline]
    fn row_bytes(&self) -> usize {
        self.row_elems() * 4
    }

    #[inline]
    fn full_mask(&self) -> u64 {
        if self.num_extract >= 64 {
            u64::MAX
        } else {
            (1u64 << self.num_extract) - 1
        }
    }

    /// Begin a prefill capture of `total_rows` positions starting at `start_abs`.
    ///
    /// Requires Idle and `start_abs == committed_abs_end`. Only the newest
    /// `min(total_rows, capacity_rows)` source rows are retained in staging.
    pub(crate) fn begin_prefill(
        &mut self,
        start_abs: usize,
        total_rows: usize,
    ) -> Result<(), String> {
        self.begin_common(HiddenStageKind::Prefill, start_abs, total_rows)
    }

    /// Begin a verify capture of `total_rows` positions starting at `start_abs`.
    ///
    /// Stages **all** rows (rejects `total_rows > stage_capacity_rows`). The
    /// committed ring is untouched until [`commit_verified_prefix`].
    pub(crate) fn begin_verify(
        &mut self,
        start_abs: usize,
        total_rows: usize,
    ) -> Result<(), String> {
        self.begin_common(HiddenStageKind::Verify, start_abs, total_rows)
    }

    /// Begin a single-token decode capture at `start_abs`.
    pub(crate) fn begin_decode(&mut self, start_abs: usize) -> Result<(), String> {
        self.begin_common(HiddenStageKind::Decode, start_abs, 1)
    }

    fn begin_common(
        &mut self,
        kind: HiddenStageKind,
        start_abs: usize,
        total_rows: usize,
    ) -> Result<(), String> {
        self.require_not_poisoned()?;
        if !matches!(self.stage, HiddenStageState::Idle) {
            return Err("glimmer hidden log: begin requires Idle stage".into());
        }
        if start_abs != self.committed_abs_end {
            return Err(format!(
                "glimmer hidden log: begin start_abs {start_abs} != committed_abs_end {}",
                self.committed_abs_end
            ));
        }
        if total_rows == 0 {
            return Err("glimmer hidden log: begin total_rows == 0".into());
        }
        let (stored_row0, stored_rows) = match kind {
            HiddenStageKind::Prefill => hidden_prefill_retention(total_rows, self.capacity_rows),
            HiddenStageKind::Verify => {
                if total_rows > self.stage_capacity_rows {
                    return Err(format!(
                        "glimmer hidden log: verify total_rows {total_rows} > stage_capacity {}",
                        self.stage_capacity_rows
                    ));
                }
                (0, total_rows)
            }
            HiddenStageKind::Decode => {
                debug_assert_eq!(total_rows, 1);
                (0, 1)
            }
        };
        if stored_rows > self.stage_capacity_rows {
            return Err(format!(
                "glimmer hidden log: stored_rows {stored_rows} > stage_capacity {}",
                self.stage_capacity_rows
            ));
        }
        self.audit_host = None;
        self.stage = HiddenStageState::Writing(HiddenStage {
            kind,
            start_abs,
            total_rows,
            stored_row0,
            stored_rows,
            captured_mask: 0,
        });
        Ok(())
    }

    /// Scatter one extract-slot's rows from `x` into staging.
    ///
    /// `x` is F32 with at least `source_rows * hidden` elements (row-major).
    /// `source_rows` must equal the transaction's `total_rows`. `ci` is the
    /// extract ordinal (0..num_extract) and must not already be captured.
    /// Only retained source rows (`stored_row0..stored_row0+stored_rows`) are
    /// copied, each via one ordered async D2D.
    pub(crate) fn scatter_layer_rows(
        &mut self,
        gpu: &Gpu,
        ci: usize,
        x: &GpuTensor,
        source_rows: usize,
    ) -> Result<(), String> {
        self.require_not_poisoned()?;
        let st = match &self.stage {
            HiddenStageState::Writing(s) => s.clone(),
            _ => return Err("glimmer hidden log: scatter requires Writing stage".into()),
        };
        if source_rows != st.total_rows {
            return Err(format!(
                "glimmer hidden log: scatter source_rows {source_rows} != total_rows {}",
                st.total_rows
            ));
        }
        if ci >= self.num_extract {
            return Err(format!(
                "glimmer hidden log: scatter ci {ci} out of range (ne={})",
                self.num_extract
            ));
        }
        if st.captured_mask & (1u64 << ci) != 0 {
            return Err(format!("glimmer hidden log: scatter duplicate ci {ci}"));
        }
        if x.dtype != DType::F32 {
            return Err("glimmer hidden log: scatter x must be F32".into());
        }
        let need = source_rows
            .checked_mul(self.hidden)
            .ok_or_else(|| "glimmer hidden log: scatter size overflow".to_string())?;
        if x.numel() < need {
            return Err(format!(
                "glimmer hidden log: scatter x numel {} < need {need}",
                x.numel()
            ));
        }

        let hidden = self.hidden;
        let ne = self.num_extract;
        let elem_bytes = 4usize;
        let row_src_bytes = hidden * elem_bytes;

        for src_row in st.stored_row0..(st.stored_row0 + st.stored_rows) {
            let stage_row = src_row - st.stored_row0;
            let src_off = src_row
                .checked_mul(row_src_bytes)
                .ok_or_else(|| "glimmer hidden log: scatter src offset overflow".to_string())?;
            let dst_elem = hidden_stage_elem_offset(stage_row, ci, 0, ne, hidden);
            let dst_off = dst_elem
                .checked_mul(elem_bytes)
                .ok_or_else(|| "glimmer hidden log: scatter dst offset overflow".to_string())?;
            gpu.memcpy_dtod_at_ordered_async(
                &self.staging.buf,
                dst_off,
                &x.buf,
                src_off,
                row_src_bytes,
            )
            .map_err(|e| format!("glimmer hidden log: scatter D2D ci={ci} row={src_row}: {e:?}"))?;
        }

        // Mark captured only after every retained row copy succeeded.
        if let HiddenStageState::Writing(s) = &mut self.stage {
            s.captured_mask |= 1u64 << ci;
        }
        Ok(())
    }

    /// Commit a completed prefill (or decode) stage into the ring and return to Idle.
    ///
    /// Requires every extract slot captured. On D2D error the stage is left
    /// non-Idle (and may be poisoned) so the caller can abort/reset.
    pub(crate) fn finish_prefill_and_commit(&mut self, gpu: &Gpu) -> Result<(), String> {
        self.require_not_poisoned()?;
        let st = match &self.stage {
            HiddenStageState::Writing(s)
                if matches!(s.kind, HiddenStageKind::Prefill | HiddenStageKind::Decode) =>
            {
                s.clone()
            }
            HiddenStageState::Writing(s) => {
                return Err(format!(
                    "glimmer hidden log: finish_prefill_and_commit got kind {:?}",
                    s.kind
                ));
            }
            _ => {
                return Err(
                    "glimmer hidden log: finish_prefill_and_commit requires Writing Prefill/Decode"
                        .into(),
                );
            }
        };
        self.require_full_mask(&st)?;
        // Prefill/decode staging holds the retained suffix at indices [0, stored_rows).
        // Absolute positions for those rows are [start+stored_row0, start+total).
        let new_end = st.start_abs + st.total_rows;
        self.commit_staging_to_ring(gpu, /*staging_idx0=*/ 0, st.stored_rows, new_end)?;
        self.stage = HiddenStageState::Idle;
        Ok(())
    }

    /// Decode commit — alias of [`finish_prefill_and_commit`] for call-site clarity.
    pub(crate) fn finish_decode_and_commit(&mut self, gpu: &Gpu) -> Result<(), String> {
        self.finish_prefill_and_commit(gpu)
    }

    /// Mark a fully-scattered verify transaction as VerifyReady without touching
    /// the committed ring interval.
    pub(crate) fn finish_verify(&mut self) -> Result<(), String> {
        self.require_not_poisoned()?;
        let st = match &self.stage {
            HiddenStageState::Writing(s) if s.kind == HiddenStageKind::Verify => s.clone(),
            HiddenStageState::Writing(s) => {
                return Err(format!(
                    "glimmer hidden log: finish_verify got kind {:?}",
                    s.kind
                ));
            }
            _ => {
                return Err("glimmer hidden log: finish_verify requires Writing Verify".into());
            }
        };
        self.require_full_mask(&st)?;
        self.stage = HiddenStageState::VerifyReady(st);
        Ok(())
    }

    /// Commit the accepted verify prefix of `keep_rows` into the ring.
    ///
    /// Requires VerifyReady and `keep_rows <= total_rows`. Rejected suffix
    /// never enters `rows`. `keep_rows == 0` advances nothing and returns Idle.
    pub(crate) fn commit_verified_prefix(
        &mut self,
        gpu: &Gpu,
        keep_rows: usize,
    ) -> Result<(), String> {
        self.require_not_poisoned()?;
        let st = match &self.stage {
            HiddenStageState::VerifyReady(s) => s.clone(),
            _ => {
                return Err(
                    "glimmer hidden log: commit_verified_prefix requires VerifyReady".into(),
                );
            }
        };
        if keep_rows > st.total_rows {
            return Err(format!(
                "glimmer hidden log: keep_rows {keep_rows} > total_rows {}",
                st.total_rows
            ));
        }
        if keep_rows == 0 {
            self.stage = HiddenStageState::Idle;
            self.audit_host = None;
            return Ok(());
        }
        let (src_row0, stored_rows) = hidden_verify_retention(keep_rows, self.capacity_rows);
        // Staging holds the full verify block at rows [0, total).
        // Copy staging[src_row0 .. src_row0+stored_rows) → abs [start+keep-stored, start+keep).
        let new_end = st.start_abs + keep_rows;
        self.commit_staging_to_ring(gpu, src_row0, stored_rows, new_end)?;
        self.stage = HiddenStageState::Idle;

        // Audit: compare committed accepted prefix (retained suffix if cap-trimmed)
        // against the stashed host shadow when present. Mismatch after publish
        // poisons — watermarks already advanced and may not match reality.
        if let Some(host) = self.audit_host.take() {
            let host_off = src_row0 * self.row_elems();
            let host_len = stored_rows * self.row_elems();
            if host.len() < host_off + host_len {
                self.poisoned = true;
                return Err(format!(
                    "glimmer device-capture audit verify_post_commit_ring: host len {} < need {}; \
                     session requires reset",
                    host.len(),
                    host_off + host_len
                ));
            }
            let abs_start = new_end - stored_rows;
            if let Err(e) = self.audit_compare_committed_rows(
                gpu,
                "verify_post_commit_ring",
                &host[host_off..host_off + host_len],
                abs_start,
                stored_rows,
            ) {
                self.poisoned = true;
                return Err(format!("{e}; session requires reset"));
            }
        }
        Ok(())
    }

    fn require_full_mask(&self, st: &HiddenStage) -> Result<(), String> {
        let full = self.full_mask();
        if st.captured_mask & full != full {
            return Err(format!(
                "glimmer hidden log: incomplete capture mask {:#x} (need {:#x})",
                st.captured_mask, full
            ));
        }
        Ok(())
    }

    /// Copy `stored_rows` contiguous staging rows starting at `staging_idx0`
    /// into the ring covering absolute `[new_end - stored_rows, new_end)`, then
    /// publish `committed_abs_end = new_end` and
    /// `valid_abs_start = new_end.saturating_sub(capacity)`.
    ///
    /// On copy error leaves watermarks unpublished and sets poison so the
    /// caller must reset; do not treat prior watermarks as trustworthy.
    fn commit_staging_to_ring(
        &mut self,
        gpu: &Gpu,
        staging_idx0: usize,
        stored_rows: usize,
        new_end: usize,
    ) -> Result<(), String> {
        if stored_rows == 0 {
            // No D2D; publishing watermarks is still atomic and safe.
            self.committed_abs_end = new_end;
            self.valid_abs_start = new_end.saturating_sub(self.capacity_rows);
            return Ok(());
        }
        let abs_start = new_end - stored_rows;
        let abs_end = new_end;
        let row_bytes = self.row_bytes();
        let ne = self.num_extract;
        let h = self.hidden;
        let cap = self.capacity_rows;

        let mut enqueued_segments = 0usize;
        for (row0, slot0, len) in hipfire_runtime::dflash::ring_segments(abs_start, abs_end, cap) {
            let seg_stage0 = staging_idx0 + (row0 - abs_start);
            let src_off = seg_stage0
                .checked_mul(row_bytes)
                .ok_or_else(|| "glimmer hidden log: commit src overflow".to_string())?;
            let dst_elem = (slot0 * ne) * h;
            let dst_off = dst_elem
                .checked_mul(4)
                .ok_or_else(|| "glimmer hidden log: commit dst overflow".to_string())?;
            let nbytes = len
                .checked_mul(row_bytes)
                .ok_or_else(|| "glimmer hidden log: commit size overflow".to_string())?;
            if let Err(e) = gpu.memcpy_dtod_at_ordered_async(
                &self.rows.buf,
                dst_off,
                &self.staging.buf,
                src_off,
                nbytes,
            ) {
                // Failure after any prior segment may leave the ring
                // physically inconsistent. Poison so callers fail-closed
                // and do not publish logical watermarks.
                self.poisoned = true;
                return Err(format!(
                    "glimmer hidden log: commit D2D slot={slot0} len={len} failed after \
                     {enqueued_segments} segment(s) enqueued; session requires reset: {e:?}"
                ));
            }
            enqueued_segments += 1;
        }

        // Publish watermarks only after every segment enqueued successfully.
        self.committed_abs_end = new_end;
        self.valid_abs_start = new_end.saturating_sub(self.capacity_rows);
        Ok(())
    }

    /// Drop the open transaction without touching the committed interval.
    /// Does **not** clear poison — only [`reset`] restores a poisoned log.
    pub(crate) fn abort_stage(&mut self) {
        self.stage = HiddenStageState::Idle;
        self.audit_host = None;
    }

    /// Reset committed interval, stage, poison, and audit shadow (no GPU memset).
    pub fn reset(&mut self) {
        self.valid_abs_start = 0;
        self.committed_abs_end = 0;
        self.stage = HiddenStageState::Idle;
        self.poisoned = false;
        self.audit_host = None;
    }

    /// Mark the log poisoned after a post-publish audit failure (or similar).
    /// Callers must reset the session before further capture ops.
    pub(crate) fn poison_for_audit(&mut self) {
        self.poisoned = true;
        self.audit_host = None;
    }

    /// Copy `n_rows` committed positions starting at `abs_start` into a contiguous
    /// destination tensor at `dst_row_offset`, using ≤2 ordered async D2Ds.
    ///
    /// Requires Idle and `[abs_start, abs_start+n_rows) ⊆ [valid_abs_start, committed_abs_end)`.
    pub fn copy_committed_rows_to(
        &self,
        gpu: &Gpu,
        abs_start: usize,
        n_rows: usize,
        dst: &GpuTensor,
        dst_row_offset: usize,
    ) -> Result<(), String> {
        self.require_not_poisoned()?;
        if !matches!(self.stage, HiddenStageState::Idle) {
            return Err("glimmer hidden log: copy_committed_rows_to requires Idle".into());
        }
        if n_rows == 0 {
            return Ok(());
        }
        let abs_end = abs_start
            .checked_add(n_rows)
            .ok_or_else(|| "glimmer hidden log: copy range overflow".to_string())?;
        if abs_start < self.valid_abs_start || abs_end > self.committed_abs_end {
            return Err(format!(
                "glimmer hidden log: copy [{abs_start},{abs_end}) outside valid [{},{})",
                self.valid_abs_start, self.committed_abs_end
            ));
        }
        if dst.dtype != DType::F32 {
            return Err("glimmer hidden log: copy dst must be F32".into());
        }
        let row_elems = self.row_elems();
        let row_bytes = self.row_bytes();
        let dst_need_elems = (dst_row_offset + n_rows)
            .checked_mul(row_elems)
            .ok_or_else(|| "glimmer hidden log: copy dst size overflow".to_string())?;
        if dst.numel() < dst_need_elems {
            return Err(format!(
                "glimmer hidden log: copy dst numel {} < need {dst_need_elems}",
                dst.numel()
            ));
        }

        let ne = self.num_extract;
        let h = self.hidden;
        let cap = self.capacity_rows;

        for (_row0, slot0, len) in hipfire_runtime::dflash::ring_segments(abs_start, abs_end, cap) {
            let src_elem = (slot0 * ne) * h;
            let src_off = src_elem * 4;
            let dst_row = dst_row_offset + (_row0 - abs_start);
            let dst_off = dst_row
                .checked_mul(row_bytes)
                .ok_or_else(|| "glimmer hidden log: copy dst offset overflow".to_string())?;
            let nbytes = len
                .checked_mul(row_bytes)
                .ok_or_else(|| "glimmer hidden log: copy size overflow".to_string())?;
            gpu.memcpy_dtod_at_ordered_async(&dst.buf, dst_off, &self.rows.buf, src_off, nbytes)
                .map_err(|e| {
                    format!("glimmer hidden log: copy D2D slot={slot0} len={len}: {e:?}")
                })?;
        }
        Ok(())
    }

    /// Whether `rewind_to(end)` is safe given the retained ring window.
    pub fn can_rewind_to(&self, end: usize) -> bool {
        !self.poisoned
            && matches!(self.stage, HiddenStageState::Idle)
            && hidden_can_rewind_to(
                end,
                self.committed_abs_end,
                self.valid_abs_start,
                self.capacity_rows,
            )
    }

    /// Publish `committed_abs_end = end` and
    /// `valid_abs_start = end.saturating_sub(capacity_rows)`.
    ///
    /// Refuses rewinds that would need an already-evicted row.
    pub fn rewind_to(&mut self, end: usize) -> Result<(), String> {
        self.require_not_poisoned()?;
        if !matches!(self.stage, HiddenStageState::Idle) {
            return Err("glimmer hidden log: rewind_to requires Idle".into());
        }
        if !hidden_can_rewind_to(
            end,
            self.committed_abs_end,
            self.valid_abs_start,
            self.capacity_rows,
        ) {
            return Err(format!(
                "glimmer hidden log: cannot rewind to {end} (committed={}, valid_start={}, cap={})",
                self.committed_abs_end, self.valid_abs_start, self.capacity_rows
            ));
        }
        self.committed_abs_end = end;
        self.valid_abs_start = end.saturating_sub(self.capacity_rows);
        Ok(())
    }

    #[inline]
    fn require_not_poisoned(&self) -> Result<(), String> {
        if self.poisoned {
            Err("glimmer hidden log: poisoned after partial device commit; \
                 session requires reset"
                .into())
        } else {
            Ok(())
        }
    }

    /// Stash host position-major capture for a later post-commit ring audit
    /// (verify path). No-op storage when `host` is empty.
    pub(crate) fn stash_audit_host(&mut self, host: Vec<f32>) {
        self.audit_host = if host.is_empty() { None } else { Some(host) };
    }

    /// Download staging `[0, stored_rows)` and bit-compare against host
    /// position-major rows `[stored_row0, stored_row0+stored_rows)`.
    ///
    /// `host` is full transaction layout `[total_rows][ne][hidden]`.
    pub(crate) fn audit_compare_staging_to_host(
        &self,
        gpu: &Gpu,
        phase: &str,
        host: &[f32],
        start_abs: usize,
        stored_row0: usize,
        stored_rows: usize,
    ) -> Result<(), String> {
        if stored_rows == 0 {
            return Ok(());
        }
        let ne = self.num_extract;
        let h = self.hidden;
        let row_elems = self.row_elems();
        let need_host = (stored_row0 + stored_rows)
            .checked_mul(row_elems)
            .ok_or_else(|| "glimmer audit: host size overflow".to_string())?;
        if host.len() < need_host {
            return Err(format!(
                "glimmer device-capture audit {phase}: host len {} < need {need_host}",
                host.len()
            ));
        }
        // download_f32 synchronizes outstanding work on the buffer.
        let staging_full = gpu
            .download_f32(&self.staging)
            .map_err(|e| format!("glimmer device-capture audit {phase} download staging: {e:?}"))?;
        let stage_need = stored_rows * row_elems;
        if staging_full.len() < stage_need {
            return Err(format!(
                "glimmer device-capture audit {phase}: staging numel {} < need {stage_need}",
                staging_full.len()
            ));
        }
        let host_slice = &host[stored_row0 * row_elems..need_host];
        let stage_slice = &staging_full[..stage_need];
        audit_compare_position_major(
            phase,
            host_slice,
            stage_slice,
            stored_rows,
            ne,
            h,
            start_abs + stored_row0,
            &self.capture_layers,
        )
    }

    /// Download the ring and bit-compare `n_rows` committed absolute positions
    /// starting at `abs_start` against contiguous host position-major data.
    pub(crate) fn audit_compare_committed_rows(
        &self,
        gpu: &Gpu,
        phase: &str,
        host: &[f32],
        abs_start: usize,
        n_rows: usize,
    ) -> Result<(), String> {
        if n_rows == 0 {
            return Ok(());
        }
        let ne = self.num_extract;
        let h = self.hidden;
        let cap = self.capacity_rows;
        let row_elems = self.row_elems();
        let need = n_rows
            .checked_mul(row_elems)
            .ok_or_else(|| "glimmer audit: committed compare size overflow".to_string())?;
        if host.len() < need {
            return Err(format!(
                "glimmer device-capture audit {phase}: host len {} < need {need}",
                host.len()
            ));
        }
        let ring = gpu
            .download_f32(&self.rows)
            .map_err(|e| format!("glimmer device-capture audit {phase} download ring: {e:?}"))?;
        for row in 0..n_rows {
            let abs = abs_start + row;
            let slot = abs % cap;
            for ci in 0..ne {
                let layer = self.capture_layers.get(ci).copied().unwrap_or(usize::MAX);
                for d in 0..h {
                    let host_off = (row * ne + ci) * h + d;
                    let ring_off = (slot * ne + ci) * h + d;
                    if ring_off >= ring.len() {
                        return Err(format!(
                            "glimmer device-capture audit {phase}: ring off {ring_off} out of {}",
                            ring.len()
                        ));
                    }
                    let e = host[host_off];
                    let a = ring[ring_off];
                    if e.to_bits() != a.to_bits() {
                        return Err(audit_bits_mismatch(
                            phase,
                            abs,
                            ci,
                            layer,
                            d,
                            e.to_bits(),
                            a.to_bits(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Free both GPU tensors. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.rows);
        let _ = gpu.free_tensor(self.staging);
    }
}

/// Validate capture-layer contract: non-empty, strictly increasing, in range,
/// and `num_extract <= 64` (mask width).
fn validate_capture_layers(capture_layers: &[usize], n_layers: usize) -> Result<(), String> {
    if capture_layers.is_empty() {
        return Err("glimmer hidden log: capture_layers empty".into());
    }
    if capture_layers.len() > 64 {
        return Err(format!(
            "glimmer hidden log: num_extract {} > 64 (mask width)",
            capture_layers.len()
        ));
    }
    let mut prev: Option<usize> = None;
    for &l in capture_layers {
        if l >= n_layers {
            return Err(format!(
                "glimmer hidden log: capture layer {l} out of range (n_layers={n_layers})"
            ));
        }
        if let Some(p) = prev {
            if l <= p {
                return Err(format!(
                    "glimmer hidden log: capture_layers not strictly increasing ({p} then {l})"
                ));
            }
        }
        prev = Some(l);
    }
    Ok(())
}

/// Upper bound on the DFlash speculation block, used to size `logits_batch`
/// once at init. The Glimmer assistant's trained `block_size` is 16.
pub const GLIMMER_MAX_SPEC_BLOCK: usize = 32;

// ───────────────────────── HFQ load helpers ─────────────────────────

fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer: tensor not found: {name}"))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if expected_n != 0 && n != expected_n {
        return Err(format!(
            "glimmer: shape mismatch for {name}: expected {expected_n}, got {n}"
        ));
    }
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        qt => {
            return Err(format!(
                "glimmer: unexpected quant_type {qt} for f32 vec {name}"
            ))
        }
    };
    Ok(f32_data)
}

/// Load an RMSNorm weight.
///
/// Muse Glimmer uses **two different norm conventions**, and mixing them up is a
/// silent-wrongness trap. Per HF `modeling_muse_glimmer.py`:
///
/// | tensor | class | convention |
/// |---|---|---|
/// | `input_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | `post_attention_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | `pre_feedforward_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | `post_feedforward_layernorm` | `MuseGlimmerTextCenteredRMSNorm` | `x_norm * (1 + w)` |
/// | final `norm` | `MuseGlimmerRMSNorm` | plain `x_norm * w` |
/// | `qk_norm`, `embed_norm` | `MuseGlimmerRMSNorm(with_scale=False)` | scale-less |
///
/// `centered` bakes `1 + w` at load so the hot path stays on the ordinary
/// `rmsnorm_f32` kernel — no new kernel, no per-call cost.
///
/// The centered classification is corroborated by the checkpoint itself: the
/// post-norms store NEGATIVE weights (`post_attention_layernorm` −0.523, −0.480,
/// −0.237; `post_feedforward_layernorm` −0.357, −0.371, −0.192), impossible under
/// plain `x * w` since they would flip the residual's sign. Under `1 + w` they
/// become sensible scales of ~0.48–0.89. The final `norm` stores ±3.x, which is
/// only sensible under the PLAIN convention — centering it was a real bug.
///
/// Opt out of centering with `HIPFIRE_GLIMMER_NO_CENTERED_NORM=1` for A/B.
fn load_norm_with(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    dim: usize,
    centered: bool,
) -> Result<GpuTensor, String> {
    let mut f32_data = load_f32_vec(hfq, name, dim)?;
    if centered
        && std::env::var("HIPFIRE_GLIMMER_NO_CENTERED_NORM")
            .ok()
            .as_deref()
            != Some("1")
    {
        for v in f32_data.iter_mut() {
            *v += 1.0;
        }
    }
    gpu.upload_f32(&f32_data, &[dim])
        .map_err(|e| format!("glimmer: upload norm {name}: {e:?}"))
}

/// Centered (`1 + w`) norm — the four per-decoder-layer norms.
fn load_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize) -> Result<GpuTensor, String> {
    load_norm_with(hfq, gpu, name, dim, true)
}

/// Plain (`w`) norm — the final `model.language_model.norm`.
fn load_norm_plain(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    dim: usize,
) -> Result<GpuTensor, String> {
    load_norm_with(hfq, gpu, name, dim, false)
}

/// quant_type → DType mapping for projection weights.
/// F16 is dequantized to F32 on upload.
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer: tensor not found: {name}"))?;
    if info.quant_type == 1 {
        let f32_data: Vec<f32> = data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let buf = gpu
            .upload_f32(&f32_data, &[m, k])
            .map_err(|e| format!("glimmer: upload F32 {name}: {e:?}"))?;
        return Ok(WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        });
    }
    let dtype = match info.quant_type {
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let buf = gpu
                .upload_f32(&f32_data, &[m, k])
                .map_err(|e| format!("glimmer: upload F32 {name}: {e:?}"))?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            });
        }
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        11 => DType::HFQ3G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        19 => DType::MQ4G256,
        qt => return Err(format!("glimmer: unsupported quant_type {qt} for {name}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("glimmer: upload {name}: {e:?}"))?;
    let awq_scale = if dtype.supports_awq_sidecar() {
        hipfire_runtime::hfq::load_awq_scale(hfq, gpu, name, k)
    } else {
        None
    };
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale,
    })
}

// ──────────────────────────── Weights ────────────────────────────

/// Per-layer weights. Glimmer uses uniform head_dim 128, so a single struct
/// covers both sliding and full layers (unlike Gemma4's Sliding/Full split).
pub struct GlimmerLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    /// Gated attention gate: `self_attn.gate_proj` — NOT the MLP gate.
    pub attn_gate_proj: WeightTensor,
    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub gate_proj: WeightTensor, // mlp.gate_proj
    pub up_proj: WeightTensor,   // mlp.up_proj
    pub down_proj: WeightTensor, // mlp.down_proj
}

pub struct GlimmerWeights {
    /// Token embedding [vocab, dim]
    pub embed_tokens: GpuTensor,
    pub embd_format: EmbeddingFormat,
    /// LM head — separate allocation (untied). NOT an alias of embed_tokens.
    pub lm_head: WeightTensor,
    pub final_norm: GpuTensor,
    pub layers: Vec<GlimmerLayerWeights>,
}

impl GlimmerWeights {
    pub fn load(hfq: &HfqFile, cfg: &GlimmerConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let dim = cfg.dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hidden_dim = cfg.hidden_dim;

        // ── Embedding ──────────────────────────────────────────────────
        let embed_name = "model.language_model.embed_tokens.weight";
        let (embed_info, embed_data) = hfq
            .tensor_data(embed_name)
            .ok_or_else(|| "glimmer: embed_tokens not found in HFQ".to_string())?;
        let (embed_tokens, embd_format) = match embed_info.quant_type {
            3 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("glimmer: upload embed: {e:?}"))?,
                EmbeddingFormat::Q8_0,
            ),
            6 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("glimmer: upload embed: {e:?}"))?,
                EmbeddingFormat::HFQ4G256,
            ),
            7 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("glimmer: upload embed: {e:?}"))?,
                EmbeddingFormat::HFQ4G128,
            ),
            1 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, dim])
                        .map_err(|e| format!("glimmer: upload embed f32: {e:?}"))?,
                    EmbeddingFormat::F32,
                )
            }
            2 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, dim])
                        .map_err(|e| format!("glimmer: upload embed f32: {e:?}"))?,
                    EmbeddingFormat::F32,
                )
            }
            qt => return Err(format!("glimmer: unsupported embed quant_type {qt}")),
        };

        // ── Untied LM head ─────────────────────────────────────────────
        // Separate tensor "lm_head.weight" [vocab, dim], not alias of embed.
        let lm_head = load_wt(hfq, gpu, "lm_head.weight", cfg.vocab_size, dim)?;

        // PLAIN norm: HF builds the final `norm` as MuseGlimmerRMSNorm
        // (with_scale=True), NOT MuseGlimmerTextCenteredRMSNorm. Centering it
        // shifts every logit and was a real bring-up bug.
        let final_norm = load_norm_plain(hfq, gpu, "model.language_model.norm.weight", dim)?;

        // ── Layers ─────────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.language_model.layers.{i}");
            layers.push(GlimmerLayerWeights {
                input_layernorm: load_norm(hfq, gpu, &format!("{p}.input_layernorm.weight"), dim)?,
                post_attention_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    dim,
                )?,
                pre_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    dim,
                )?,
                post_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    dim,
                )?,
                attn_gate_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    q_dim,
                    dim,
                )?,
                q_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    dim,
                )?,
                k_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    dim,
                )?,
                v_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    dim,
                )?,
                o_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    dim,
                    q_dim,
                )?,
                gate_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    hidden_dim,
                    dim,
                )?,
                up_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    hidden_dim,
                    dim,
                )?,
                down_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    dim,
                    hidden_dim,
                )?,
            });
        }

        Ok(GlimmerWeights {
            embed_tokens,
            embd_format,
            lm_head,
            final_norm,
            layers,
        })
    }

    /// Return all GPU weight buffers to the pool. Consumes self.
    /// lm_head is a separate allocation (untied) and IS freed here,
    /// unlike Gemma4's tied alias.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        self.lm_head.free_all(gpu);
        let _ = gpu.free_tensor(self.final_norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_layernorm);
            let _ = gpu.free_tensor(l.post_attention_layernorm);
            let _ = gpu.free_tensor(l.pre_feedforward_layernorm);
            let _ = gpu.free_tensor(l.post_feedforward_layernorm);
            l.attn_gate_proj.free_all(gpu);
            l.q_proj.free_all(gpu);
            l.k_proj.free_all(gpu);
            l.v_proj.free_all(gpu);
            l.o_proj.free_all(gpu);
            l.gate_proj.free_all(gpu);
            l.up_proj.free_all(gpu);
            l.down_proj.free_all(gpu);
        }
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV caches.
///
/// # KV cache topology
/// Head_dim is uniform 128 (`lib.rs` — unlike Gemma4's 256/512 split), so
/// a single cache shape would suffice physically. We retain the dual-cache
/// topology (sliding + full) for logical isolation: sliding layers attend
/// with a 2048 window while full layers attend full-causal (window=0). The
/// two caches share identical geometry `(head_dim=128, n_kv=2)` and differ
/// only in layer count (39 vs 13) and window at attention dispatch. A single
/// unified cache would also be valid and save one bookkeeping vector; we
/// keep the split to mirror Gemma4's proven eviction/windowing discipline
/// without paying a shape-mismatch cost.
pub struct GlimmerState {
    /// Sliding-window KV cache (head_dim 128), one slot per sliding layer.
    pub kv_sliding: KvCache,
    /// Full-attention KV cache (head_dim 128), one slot per full layer.
    pub kv_full: KvCache,
    /// Per-layer slot index into the matching per-type cache.
    pub kv_slot_for_layer: Vec<usize>,

    pub pos_buf: hip_bridge::DeviceBuffer,
    /// Stable host source for the device position scalar (hipGraph-safe if ever
    /// captured: the captured memcpy re-reads this Box on replay).
    pub pos_host: Box<[i32]>,
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual stream + scratch
    pub x: GpuTensor,        // [dim]
    pub residual: GpuTensor, // [dim]
    pub tmp: GpuTensor,      // [dim] norm scratch
    pub x_rot: GpuTensor,    // [dim] FWHT scratch for shared rotation (MQ4)

    // attention scratch (uniform dims)
    pub q: GpuTensor,         // [q_dim]
    pub k: GpuTensor,         // [kv_dim]
    pub v: GpuTensor,         // [kv_dim]
    pub attn_out: GpuTensor,  // [q_dim]
    pub attn_gate: GpuTensor, // [q_dim] sigmoid gate

    /// Ones-filled weight for scale-less QK-norm (head_dim ones).
    pub qk_norm_ones: GpuTensor, // [head_dim]
    /// Ones-filled weight for the scale-less embedding norm (hidden_size ones).
    ///
    /// HF wraps the embedding table in `MuseGlimmerTextNormedEmbedding`, whose
    /// `forward` is `embed_norm(Embedding::forward(ids))` with
    /// `MuseGlimmerRMSNorm(eps=rms_norm_eps, with_scale=False)`. The norm is
    /// deliberately NOT folded into the embedding matrix upstream ("Dflash
    /// implem needs to embed without the norm"), so it must run per lookup.
    pub embed_norm_ones: GpuTensor, // [hidden_size]

    // FFN scratch
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // head
    pub logits: GpuTensor, // [vocab]
    /// Scratch for on-device sampling: holds `[sampled_token_u32, new_rng_u32]`
    /// so AR decode pays an 8-byte D2H instead of `vocab_size * 4` bytes
    /// (~808 KB at 202048) per token.
    pub sample_out: GpuTensor, // [2] of u32
    /// Persistent [block_max * vocab] logits buffer for the batched lm_head.
    ///
    /// Allocated ONCE. The batched lm_head previously did `alloc_tensor` +
    /// `free_tensor` of this ~12.9 MB buffer on every call, and a cold
    /// `hipMalloc` is both slow and synchronizing — which is what made the
    /// FIRST batched lm_head of each window (the draft's) cost 69 ms while the
    /// SECOND (verify's, reusing the block the first had just freed) cost 7.6 ms
    /// for the same weight through the same kernel.
    pub logits_batch: GpuTensor, // [GLIMMER_MAX_SPEC_BLOCK * vocab]
    /// Persistent [block_max] argmax indices for the batched lm_head path.
    /// F32 slots, consumed as i32 output (same packing as other argmax buffers).
    pub argmax_batch: GpuTensor, // [GLIMMER_MAX_SPEC_BLOCK]
    /// Batched flash-attention partials for prefill over-window recovery.
    /// Lazily allocated on first over-window sliding chunk: n_heads *
    /// ceil(max_seq/128) * (2+head_dim) * 64 floats (~65 MiB at max_seq=8192).
    /// Factor-64 precedent: crates/hipfire-arch-cohere2moe/src/cohere2moe.rs:496-511.
    pub prefill_flash_partials: Option<GpuTensor>,
    /// Single-element i32 position tensor for the flash decode path. `pos_buf`
    /// is a raw DeviceBuffer, but the batched flash kernel takes a GpuTensor of
    /// positions; at batch_size=1 it holds the same value. Lazily allocated.
    pub decode_pos: Option<GpuTensor>,
    /// Optional device-resident DFlash target-hidden ring. `None` keeps the
    /// host-capture path unchanged. Owned here so free/reset stay with target KV.
    pub(crate) target_hidden_log: Option<GlimmerHiddenLog>,
}

impl GlimmerState {
    pub fn new(gpu: &mut Gpu, cfg: &GlimmerConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let dim = cfg.dim;

        // Two Q8 KV caches: one slot per layer of the matching type.
        // Both have identical head_dim=128 geometry; split is logical.
        //
        // VMM backend by default. The contiguous constructor allocates every
        // slot of max_seq for all 52 layers up front, which is what caps
        // context on a 16 GB card: weights are 15.5 GB and KV is nearly the
        // whole remainder (52 x max_seq x 544 B = 3.71 GB at 131072, against
        // ~490 MB of headroom). The VMM path reserves VIRTUAL address space
        // and maps physical pages only as the context actually reaches them
        // (alloc_vmm_tensor with initial_mapped_bytes = 0, then
        // ensure_mapped_capacity as n_tokens grows).
        //
        // Crucially this needs NO kernel change: the reserve is one linear
        // virtual range, so `k_cache + t * stride` and `scores[t]` keep
        // indexing absolutely. A window-sized ring buffer would have required
        // wraparound in every consumer; this does not.
        //
        // Precedent: DeepSeek4 (hipfire-arch-deepseek4/src/forward.rs:1813).
        // `HIPFIRE_GLIMMER_KV_VMM=0` falls back to the contiguous allocator.
        let use_vmm = std::env::var("HIPFIRE_GLIMMER_KV_VMM")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true);
        let (kv_sliding, kv_full) = if use_vmm {
            let sliding_layers = vec![true; cfg.n_sliding_layers()];
            let full_layers = vec![true; cfg.n_full_layers()];
            let s = KvCache::new_gpu_q8_vmm_capped_filtered(
                gpu,
                &sliding_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                max_seq,
                max_seq,
            )
            .map_err(|e| format!("glimmer: sliding kv cache (vmm): {e:?}"))?;
            let f = KvCache::new_gpu_q8_vmm_capped_filtered(
                gpu,
                &full_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                max_seq,
                max_seq,
            )
            .map_err(|e| format!("glimmer: full kv cache (vmm): {e:?}"))?;
            (s, f)
        } else {
            let s = KvCache::new_gpu_q8(
                gpu,
                cfg.n_sliding_layers(),
                cfg.n_kv_heads,
                cfg.head_dim,
                max_seq,
            )
            .map_err(|e| format!("glimmer: sliding kv cache: {e:?}"))?;
            let f = KvCache::new_gpu_q8(
                gpu,
                cfg.n_full_layers(),
                cfg.n_kv_heads,
                cfg.head_dim,
                max_seq,
            )
            .map_err(|e| format!("glimmer: full kv cache: {e:?}"))?;
            (s, f)
        };

        // Per-layer slot mapping: sequential count within each type.
        let mut kv_slot_for_layer = Vec::with_capacity(cfg.n_layers);
        let mut s = 0usize;
        let mut f = 0usize;
        for &lt in cfg.layer_types.iter() {
            match lt {
                crate::config::GlimmerLayerType::Sliding => {
                    kv_slot_for_layer.push(s);
                    s += 1;
                }
                crate::config::GlimmerLayerType::Full => {
                    kv_slot_for_layer.push(f);
                    f += 1;
                }
            }
        }

        // FWHT sign LUT must exist before any fused_rmsnorm_rotate_mq
        // launch (the shared-rotation path). Mirrors gemma4's ensure at state init.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("glimmer: ensure_mq_signs: {e:?}"))?;

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("glimmer: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("glimmer: alloc {label}: {e:?}"))
        };

        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hd = cfg.head_dim;

        // Ones for scale-less QK-norm.
        let qk_norm_ones = alloc(gpu, hd, "qk_norm_ones")?;
        {
            let ones: Vec<f32> = vec![1.0; hd];
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
            gpu.hip
                .memcpy_htod(&qk_norm_ones.buf, bytes)
                .map_err(|e| format!("glimmer: init qk_norm_ones: {e:?}"))?;
        }

        let embed_norm_ones = alloc(gpu, dim, "embed_norm_ones")?;
        {
            let ones: Vec<f32> = vec![1.0; dim];
            let bytes =
                unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
            gpu.hip
                .memcpy_htod(&embed_norm_ones.buf, bytes)
                .map_err(|e| format!("glimmer: upload embed_norm_ones: {e:?}"))?;
        }

        Ok(GlimmerState {
            kv_sliding,
            kv_full,
            kv_slot_for_layer,
            pos_buf,
            pos_host: vec![0i32; 1].into_boxed_slice(),
            max_seq,
            n_tokens: 0,
            x: alloc(gpu, dim, "x")?,
            residual: alloc(gpu, dim, "residual")?,
            tmp: alloc(gpu, dim, "tmp")?,
            x_rot: alloc(gpu, dim, "x_rot")?,
            q: alloc(gpu, q_dim, "q")?,
            k: alloc(gpu, kv_dim, "k")?,
            v: alloc(gpu, kv_dim, "v")?,
            attn_out: alloc(gpu, q_dim, "attn_out")?,
            attn_gate: alloc(gpu, q_dim, "attn_gate")?,
            qk_norm_ones,
            embed_norm_ones,
            gate_ffn: alloc(gpu, cfg.hidden_dim, "gate_ffn")?,
            up_ffn: alloc(gpu, cfg.hidden_dim, "up_ffn")?,
            ffn_hidden: alloc(gpu, cfg.hidden_dim, "ffn_hidden")?,
            ffn_out: alloc(gpu, dim, "ffn_out")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            sample_out: alloc(gpu, 2, "sample_out")?,
            logits_batch: alloc(gpu, GLIMMER_MAX_SPEC_BLOCK * cfg.vocab_size, "logits_batch")?,
            argmax_batch: alloc(gpu, GLIMMER_MAX_SPEC_BLOCK, "argmax_batch")?,
            prefill_flash_partials: None,
            decode_pos: None,
            target_hidden_log: None,
        })
    }

    pub fn reset(&mut self) {
        self.n_tokens = 0;
        if let Some(log) = self.target_hidden_log.as_mut() {
            log.reset();
        }
    }

    /// Allocate and install a device hidden-capture log.
    ///
    /// `ctx_cap` is the drafter context capacity (clamped to `max_seq` by the
    /// caller if desired; this method rejects `ctx_cap == 0`). `capture_layers`
    /// must be non-empty, strictly increasing, and in `0..n_layers`. `hidden`
    /// must match the residual stream width. Allocation is all-or-nothing:
    /// if staging fails after `rows`, `rows` is freed and nothing is installed.
    pub fn enable_device_hidden_capture(
        &mut self,
        gpu: &mut Gpu,
        capture_layers: &[usize],
        ctx_cap: usize,
        n_layers: usize,
        hidden: usize,
    ) -> Result<(), String> {
        if self.target_hidden_log.is_some() {
            return Err("glimmer: device hidden capture already enabled".into());
        }
        if ctx_cap == 0 {
            return Err("glimmer: enable_device_hidden_capture ctx_cap == 0".into());
        }
        if hidden == 0 {
            return Err("glimmer: enable_device_hidden_capture hidden == 0".into());
        }
        validate_capture_layers(capture_layers, n_layers)?;

        let capacity_rows = ctx_cap;
        let num_extract = capture_layers.len();
        let stage_capacity_rows = GLIMMER_MAX_SPEC_BLOCK.max(capacity_rows.min(512));

        let rows_elems = capacity_rows
            .checked_mul(num_extract)
            .and_then(|v| v.checked_mul(hidden))
            .ok_or_else(|| "glimmer: hidden log rows size overflow".to_string())?;
        let staging_elems = stage_capacity_rows
            .checked_mul(num_extract)
            .and_then(|v| v.checked_mul(hidden))
            .ok_or_else(|| "glimmer: hidden log staging size overflow".to_string())?;

        let rows = gpu
            .zeros(&[rows_elems], DType::F32)
            .map_err(|e| format!("glimmer: alloc target_hidden_log.rows: {e:?}"))?;
        let staging = match gpu.zeros(&[staging_elems], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.release_tensor_immediate(rows);
                return Err(format!("glimmer: alloc target_hidden_log.staging: {e:?}"));
            }
        };

        let mut layer_to_slot = vec![None; n_layers];
        for (ci, &layer) in capture_layers.iter().enumerate() {
            layer_to_slot[layer] = Some(ci);
        }

        self.target_hidden_log = Some(GlimmerHiddenLog {
            rows,
            staging,
            capacity_rows,
            stage_capacity_rows,
            num_extract,
            hidden,
            capture_layers: capture_layers.to_vec(),
            layer_to_slot,
            valid_abs_start: 0,
            committed_abs_end: 0,
            stage: HiddenStageState::Idle,
            poisoned: false,
            audit_host: None,
        });
        Ok(())
    }

    #[inline]
    pub fn device_hidden_capture_enabled(&self) -> bool {
        self.target_hidden_log.is_some()
    }

    #[inline]
    pub fn target_hidden_log(&self) -> Option<&GlimmerHiddenLog> {
        self.target_hidden_log.as_ref()
    }

    #[inline]
    pub fn target_hidden_log_mut(&mut self) -> Option<&mut GlimmerHiddenLog> {
        self.target_hidden_log.as_mut()
    }

    /// Return all GPU state buffers to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv_sliding.free_gpu(gpu);
        let _ = self.kv_full.free_gpu(gpu);
        let _ = gpu.hip.free(self.pos_buf);
        for t in [
            self.x,
            self.residual,
            self.tmp,
            self.x_rot,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.attn_gate,
            self.qk_norm_ones,
            self.embed_norm_ones,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.sample_out,
            self.logits_batch,
            self.argmax_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.decode_pos {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.prefill_flash_partials {
            let _ = gpu.free_tensor(t);
        }
        if let Some(log) = self.target_hidden_log {
            log.free_gpu(gpu);
        }
    }
}

// ──────────────────── Hidden-log pure metadata tests ────────────────────

#[cfg(test)]
mod hidden_log_meta_tests {
    use super::*;

    #[test]
    fn prefill_retention_no_cap_trim() {
        assert_eq!(hidden_prefill_retention(8, 16), (0, 8));
        assert_eq!(hidden_prefill_retention(16, 16), (0, 16));
    }

    #[test]
    fn prefill_retention_suffix_when_b_gt_cap() {
        assert_eq!(hidden_prefill_retention(20, 16), (4, 16));
        assert_eq!(hidden_prefill_retention(512, 17), (495, 17));
    }

    #[test]
    fn verify_retention_keep_variants() {
        assert_eq!(hidden_verify_retention(1, 16), (0, 1));
        assert_eq!(hidden_verify_retention(8, 16), (0, 8));
        assert_eq!(hidden_verify_retention(16, 16), (0, 16));
        // keep > capacity: newest capacity rows of the accepted prefix
        assert_eq!(hidden_verify_retention(20, 16), (4, 16));
    }

    #[test]
    fn ring_elem_offset_matches_host_layout() {
        let cap = 8usize;
        let ne = 5usize;
        let h = 4usize;
        assert_eq!(hidden_ring_elem_offset(0, 0, 0, cap, ne, h), 0);
        assert_eq!(hidden_ring_elem_offset(0, 2, 1, cap, ne, h), 9);
        // p=9 wraps to slot 1
        assert_eq!(hidden_ring_elem_offset(9, 0, 0, cap, ne, h), 20);
        assert_eq!(
            hidden_ring_elem_offset(15, 4, 3, cap, ne, h),
            (7 * ne + 4) * h + 3
        );
    }

    #[test]
    fn stage_elem_offset_no_wrap() {
        let ne = 5usize;
        let h = 4usize;
        assert_eq!(hidden_stage_elem_offset(0, 0, 0, ne, h), 0);
        assert_eq!(
            hidden_stage_elem_offset(3, 2, 1, ne, h),
            (3 * ne + 2) * h + 1
        );
    }

    #[test]
    fn rewind_supported_short_window() {
        assert!(hidden_can_rewind_to(10, 10, 0, 16));
        assert!(hidden_can_rewind_to(5, 10, 0, 16));
        assert!(hidden_can_rewind_to(0, 10, 0, 16));
        assert!(!hidden_can_rewind_to(11, 10, 0, 16));
    }

    #[test]
    fn rewind_refuses_evicted_row() {
        // Full ring: committed=30, valid=13, cap=17.
        let committed = 30usize;
        let valid = 13usize;
        let cap = 17usize;
        assert!(hidden_can_rewind_to(30, committed, valid, cap));
        assert!(!hidden_can_rewind_to(29, committed, valid, cap));
        assert!(!hidden_can_rewind_to(20, committed, valid, cap));
        assert!(!hidden_can_rewind_to(0, committed, valid, cap));
    }

    #[test]
    fn rewind_after_partial_fill_ok() {
        assert!(hidden_can_rewind_to(20, 20, 3, 17));
        assert!(!hidden_can_rewind_to(19, 20, 3, 17));
        assert!(hidden_can_rewind_to(3, 10, 0, 17));
    }

    #[test]
    fn validate_layers_strictly_increasing() {
        assert!(validate_capture_layers(&[1, 13, 25, 37, 49], 52).is_ok());
        assert!(validate_capture_layers(&[], 52).is_err());
        assert!(validate_capture_layers(&[1, 13, 13, 37], 52).is_err());
        assert!(validate_capture_layers(&[1, 25, 13], 52).is_err());
        assert!(validate_capture_layers(&[1, 13, 52], 52).is_err());
    }

    /// Metadata-only stage machine using null tensors (no HIP).
    fn meta_log(cap: usize, stage_cap: usize, ne: usize, hidden: usize) -> GlimmerHiddenLog {
        let n_layers = 52;
        let layers: Vec<usize> = (0..ne).map(|i| 1 + i * 12).collect();
        let mut layer_to_slot = vec![None; n_layers];
        for (ci, &l) in layers.iter().enumerate() {
            layer_to_slot[l] = Some(ci);
        }
        GlimmerHiddenLog {
            rows: GpuTensor::null_for_test(),
            staging: GpuTensor::null_for_test(),
            capacity_rows: cap,
            stage_capacity_rows: stage_cap,
            num_extract: ne,
            hidden,
            capture_layers: layers,
            layer_to_slot,
            valid_abs_start: 0,
            committed_abs_end: 0,
            stage: HiddenStageState::Idle,
            poisoned: false,
            audit_host: None,
        }
    }

    #[test]
    fn begin_requires_idle_and_append_cursor() {
        let mut log = meta_log(16, 32, 5, 8);
        assert!(log.begin_prefill(0, 4).is_ok());
        assert!(!log.stage_is_idle());
        assert!(log.begin_prefill(0, 1).is_err()); // not Idle
        log.abort_stage();
        assert!(log.begin_prefill(1, 1).is_err()); // start != committed end
        assert!(log.begin_prefill(0, 2).is_ok());
    }

    #[test]
    fn prefill_stores_suffix_only() {
        let mut log = meta_log(16, 32, 5, 8);
        log.begin_prefill(0, 20).unwrap();
        match log.stage() {
            HiddenStageState::Writing(s) => {
                assert_eq!(s.kind, HiddenStageKind::Prefill);
                assert_eq!(s.total_rows, 20);
                assert_eq!(s.stored_row0, 4);
                assert_eq!(s.stored_rows, 16);
            }
            other => panic!("expected Writing, got {other:?}"),
        }
    }

    #[test]
    fn verify_rejects_over_stage_capacity() {
        let mut log = meta_log(16, 16, 5, 8);
        assert!(log.begin_verify(0, 17).is_err());
        assert!(log.begin_verify(0, 16).is_ok());
        match log.stage() {
            HiddenStageState::Writing(s) => {
                assert_eq!(s.stored_row0, 0);
                assert_eq!(s.stored_rows, 16);
            }
            other => panic!("expected Writing, got {other:?}"),
        }
    }

    #[test]
    fn finish_verify_needs_full_mask_then_ready() {
        let mut log = meta_log(16, 32, 2, 4);
        log.begin_verify(0, 8).unwrap();
        assert!(log.finish_verify().is_err());
        if let HiddenStageState::Writing(s) = &mut log.stage {
            s.captured_mask = 0b11;
        }
        assert!(log.finish_verify().is_ok());
        assert!(matches!(log.stage(), HiddenStageState::VerifyReady(_)));
        // keep=0 contract: Idle, committed end unchanged (no D2D).
        log.stage = HiddenStageState::Idle;
        assert!(log.stage_is_idle());
        assert_eq!(log.committed_abs_end(), 0);
    }

    #[test]
    fn abort_and_reset_clear_stage_and_watermarks() {
        let mut log = meta_log(16, 32, 5, 8);
        log.begin_decode(0).unwrap();
        log.abort_stage();
        assert!(log.stage_is_idle());
        log.committed_abs_end = 12;
        log.valid_abs_start = 0;
        log.reset();
        assert_eq!(log.committed_abs_end(), 0);
        assert_eq!(log.valid_abs_start(), 0);
        assert!(log.stage_is_idle());
    }

    #[test]
    fn rewind_to_metadata() {
        let mut log = meta_log(16, 32, 5, 8);
        log.committed_abs_end = 10;
        log.valid_abs_start = 0;
        assert!(log.can_rewind_to(7));
        log.rewind_to(7).unwrap();
        assert_eq!(log.committed_abs_end(), 7);
        assert_eq!(log.valid_abs_start(), 0);

        log.committed_abs_end = 30;
        log.valid_abs_start = 14; // cap 16 → 30-16=14
        assert!(!log.can_rewind_to(29));
        assert!(log.rewind_to(29).is_err());
        assert!(log.can_rewind_to(30));
        log.rewind_to(30).unwrap();
        assert_eq!(log.valid_abs_start(), 14);
    }

    #[test]
    fn rejected_suffix_does_not_advance_on_keep_zero() {
        let mut log = meta_log(16, 32, 2, 4);
        log.committed_abs_end = 5;
        log.valid_abs_start = 0;
        log.stage = HiddenStageState::VerifyReady(HiddenStage {
            kind: HiddenStageKind::Verify,
            start_abs: 5,
            total_rows: 16,
            stored_row0: 0,
            stored_rows: 16,
            captured_mask: 0b11,
        });
        // Mirror keep=0 branch of commit_verified_prefix (no D2D).
        assert!(matches!(log.stage(), HiddenStageState::VerifyReady(_)));
        log.stage = HiddenStageState::Idle;
        assert_eq!(log.committed_abs_end(), 5);
        assert!(log.stage_is_idle());
    }

    #[test]
    fn begin_decode_is_single_row() {
        let mut log = meta_log(16, 32, 5, 8);
        log.begin_decode(0).unwrap();
        match log.stage() {
            HiddenStageState::Writing(s) => {
                assert_eq!(s.kind, HiddenStageKind::Decode);
                assert_eq!(s.total_rows, 1);
                assert_eq!(s.stored_row0, 0);
                assert_eq!(s.stored_rows, 1);
            }
            other => panic!("expected Writing, got {other:?}"),
        }
    }

    #[test]
    fn stage_not_idle_rejects_rewind() {
        let mut log = meta_log(16, 32, 5, 8);
        log.committed_abs_end = 8;
        log.begin_prefill(8, 2).unwrap();
        assert!(!log.can_rewind_to(8));
        assert!(log.rewind_to(8).is_err());
    }

    #[test]
    fn poison_rejects_begin_copy_rewind_until_reset() {
        let mut log = meta_log(16, 32, 5, 8);
        log.poisoned = true;
        assert!(log.is_poisoned());
        assert!(!log.can_rewind_to(0));
        let err = log.begin_prefill(0, 1).unwrap_err();
        assert!(
            err.contains("session requires reset"),
            "begin error must mention reset: {err}"
        );
        let err = log.rewind_to(0).unwrap_err();
        assert!(err.contains("session requires reset"), "{err}");
        // reset clears poison
        log.reset();
        assert!(!log.is_poisoned());
        assert!(log.begin_prefill(0, 2).is_ok());
        log.abort_stage();
        assert!(log.can_rewind_to(0));
    }

    #[test]
    fn abort_does_not_clear_poison() {
        let mut log = meta_log(16, 32, 5, 8);
        log.poisoned = true;
        log.abort_stage();
        assert!(log.is_poisoned());
        assert!(log.stage_is_idle());
        log.reset();
        assert!(!log.is_poisoned());
    }

    #[test]
    fn audit_compare_position_major_detects_bits() {
        let layers = [1usize, 13];
        let host = vec![1.0f32, 2.0, 3.0, 4.0]; // 1 row, 2 slots, h=2
        let mut device = host.clone();
        assert!(audit_compare_position_major("t", &host, &device, 1, 2, 2, 0, &layers).is_ok());
        device[3] = f32::from_bits(0x3f80_0001); // 1.0 + 1 ulp vs 4.0
        let err =
            audit_compare_position_major("t", &host, &device, 1, 2, 2, 7, &layers).unwrap_err();
        assert!(err.contains("phase=t"));
        assert!(err.contains("abs_row=7"));
        assert!(err.contains("capture_slot=1"));
        assert!(err.contains("layer=13"));
        assert!(err.contains("dim=1"));
        assert!(err.contains("expected_bits="));
        assert!(err.contains("actual_bits="));
    }
}
