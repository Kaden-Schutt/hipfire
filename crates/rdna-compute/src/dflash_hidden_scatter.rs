// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S2 launch fusion: exact gfx1100 hidden-ring scatter kernels.
//!
//! Replaces the per-row `memcpy_dtod_at` storms in
//! `HiddenStateRingBuffer::commit_staging_to_ring` and
//! `scatter_hidden_block_to_interleaved` (both in `hipfire-arch-qwen35`'s
//! `speculative.rs`) with one kernel launch each. Specialized to the
//! measured DFlash route: `num_extract == 5`, F32, gfx1100. The five source
//! and five destination pointers travel directly in the kernarg blob — no
//! per-cycle pointer table is built, uploaded, or retained.
//!
//! Routing contract (checked in-crate so the `&Gpu` scatter path can route
//! without a signature change):
//! - [`Gpu::dflash_hidden_commit5_applicable`] is the full fused-commit
//!   predicate: gfx1100, kill switch clear, 5+5 F32 buffers with enough
//!   elements, `n <= max_pos`, and neither hipGraph capture nor retained
//!   replay recording active (the kernels bake the current head, so they
//!   must never be captured).
//! - [`Gpu::dflash_hidden_commit5_launch`] ensures BOTH kernels (commit and
//!   scatter) then launches commit5. The commit runs strictly before any
//!   same-cycle scatter, so by the time
//!   [`Gpu::dflash_hidden_scatter5_try`] runs the scatter symbol is already
//!   loaded; a scatter that arrives with no prior fused commit (seed paths,
//!   non-gfx1100, kill switch) finds the symbol missing and reports `false`
//!   so the caller runs today's loop byte-for-byte.
//! - [`Gpu::dflash_hidden_scatter5_try`] returns `Ok(true)` when it launched
//!   (or when there were zero retained rows, a no-op in both paths) and
//!   `Ok(false)` when the caller must run the loop.
//!
//! Both kernels are pure F32 copies with one writer per destination
//! element: fused output is bit-identical to the loops. `rows == 0` / `n ==
//! 0` never launches; head/written accounting stays with the caller.

use crate::Gpu;
use crate::GpuTensor;
use hip_bridge::HipResult;

pub const DFLASH_HIDDEN_SCATTER_SRC: &str =
    include_str!("../../../kernels/src/dflash_hidden_scatter.gfx1100.hip");
pub const DFLASH_HIDDEN_COMMIT5: &str = "dflash_hidden_commit5_gfx1100";
pub const DFLASH_HIDDEN_SCATTER5: &str = "dflash_hidden_scatter5_gfx1100";

const HIDDEN_SCATTER_BLOCK: u32 = 256;
/// Absolute-addressing sentinel: the loop's `dst_modulus == usize::MAX`
/// branch. Compared as u64 in the kernel.
const DST_MODULUS_ABSENT: u64 = u64::MAX;

fn all_f32(tensors: &[GpuTensor]) -> bool {
    tensors.iter().all(|t| t.dtype == crate::DType::F32)
}

impl Gpu {
    /// Full fused-commit predicate. No allocation, no host reads, no JIT —
    /// safe to evaluate on the decode hot path.
    pub fn dflash_hidden_commit5_applicable(
        &self,
        staging: &[GpuTensor],
        dst: &[GpuTensor],
        n: usize,
        hidden: usize,
        max_pos: usize,
    ) -> bool {
        if !self.arch_caps.is_gfx1100() {
            return false;
        }
        if self.flags.hidden_scatter_fuse_off {
            return false;
        }
        // Head-dependent kernargs must never be captured or recorded.
        if self.graphs.capture_mode || self.replay.is_recording() {
            return false;
        }
        if staging.len() != 5 || dst.len() != 5 {
            return false;
        }
        if hidden == 0 || max_pos == 0 {
            return false;
        }
        // Single-wrap range: the fused grid covers (head + r) % max_pos for
        // r in 0..n. Larger n would wrap twice (a second writer per element
        // in the kernel, an OOB write in the loop) — keep today's loop.
        if n > max_pos {
            return false;
        }
        if !all_f32(staging) || !all_f32(dst) {
            return false;
        }
        let row_elems = n.checked_mul(hidden);
        let ring_elems = max_pos.checked_mul(hidden);
        let (Some(row_elems), Some(ring_elems)) = (row_elems, ring_elems) else {
            return false;
        };
        if staging.iter().any(|t| t.numel() < row_elems) {
            return false;
        }
        if dst.iter().any(|t| t.numel() < ring_elems) {
            return false;
        }
        true
    }

    /// Launch commit5 after [`Gpu::dflash_hidden_commit5_applicable`].
    /// Ensures both S2 symbols (the same-cycle scatter reuses the scatter
    /// symbol without its own `&mut` ensure), then copies
    /// `staging[ext][r, :] -> dst[ext][(head + r) % max_pos, :]` in one
    /// launch. `n == 0` advances nothing and launches nothing.
    pub fn dflash_hidden_commit5_launch(
        &mut self,
        staging: &[GpuTensor],
        dst: &[GpuTensor],
        head: usize,
        n: usize,
        hidden: usize,
        max_pos: usize,
    ) -> HipResult<()> {
        assert_eq!(staging.len(), 5, "commit5 requires exactly 5 staging bufs");
        assert_eq!(dst.len(), 5, "commit5 requires exactly 5 ring bufs");
        self.bind_thread()?;
        // Ensure the scatter symbol too: the fused commit strictly precedes
        // any same-cycle scatter, so the `&Gpu` scatter path below never
        // needs its own ensure. Both are outside any capture here.
        self.ensure_kernel(
            DFLASH_HIDDEN_COMMIT5,
            DFLASH_HIDDEN_SCATTER_SRC,
            DFLASH_HIDDEN_COMMIT5,
        )?;
        self.ensure_kernel(
            DFLASH_HIDDEN_SCATTER5,
            DFLASH_HIDDEN_SCATTER_SRC,
            DFLASH_HIDDEN_SCATTER5,
        )?;
        let total: u64 = 5u64 * (n as u64) * (hidden as u64);
        if total == 0 {
            return Ok(());
        }
        debug_assert!(total <= u64::from(u32::MAX), "commit5 grid overflow");
        let grid_x = ((total + u64::from(HIDDEN_SCATTER_BLOCK) - 1)
            / u64::from(HIDDEN_SCATTER_BLOCK)) as u32;
        debug_assert!(head <= i32::MAX as usize, "commit5 head overflow");
        debug_assert!(n <= i32::MAX as usize, "commit5 n overflow");
        debug_assert!(hidden <= i32::MAX as usize, "commit5 hidden overflow");
        debug_assert!(max_pos <= i32::MAX as usize, "commit5 max_pos overflow");
        let head_i = head as i32;
        let n_i = n as i32;
        let hidden_i = hidden as i32;
        let max_pos_i = max_pos as i32;
        let mut blob = hip_bridge::KernargBlob::new();
        for t in staging {
            blob.push_ptr(t.buf.as_ptr());
        }
        for t in dst {
            blob.push_ptr(t.buf.as_ptr());
        }
        blob.push_i32(head_i);
        blob.push_i32(n_i);
        blob.push_i32(hidden_i);
        blob.push_i32(max_pos_i);
        blob.pad_to(16);
        self.launch_kernel_blob(
            DFLASH_HIDDEN_COMMIT5,
            [grid_x, 1, 1],
            [HIDDEN_SCATTER_BLOCK, 1, 1],
            0,
            blob.as_mut_slice(),
        )
    }

    /// Fused scatter attempt on a shared `&Gpu`.
    ///
    /// Copies the retained block rows (`r_skip <= r < n_rows`, ring slot
    /// `(start_slot + (r - r_skip)) % max_pos`) into
    /// `dst[((dst_row_offset + r) % dst_modulus), ext, :]`, preserving the
    /// loop's `usize::MAX` absolute-addressing branch. Returns `Ok(true)`
    /// when the kernel launched — or when there are no retained rows
    /// (`r_skip >= n_rows`), a no-op in both paths. Returns `Ok(false)`
    /// when the caller must run the loop (wrong arch, kill switch,
    /// capture/recording, non-5-extract or non-F32 shapes, undersized
    /// buffers, or symbol not yet ensured by a fused commit).
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_hidden_scatter5_try(
        &self,
        src: &[GpuTensor],
        dst: &GpuTensor,
        start_slot: usize,
        n_rows: usize,
        r_skip: usize,
        hidden: usize,
        max_pos: usize,
        dst_row_offset: usize,
        dst_modulus: usize,
        num_extract: usize,
    ) -> HipResult<bool> {
        let rows = n_rows.saturating_sub(r_skip);
        if rows == 0 {
            return Ok(true);
        }
        if !self.arch_caps.is_gfx1100() {
            return Ok(false);
        }
        if self.flags.hidden_scatter_fuse_off {
            return Ok(false);
        }
        if self.graphs.capture_mode || self.replay.is_recording() {
            return Ok(false);
        }
        if src.len() != 5 || num_extract != 5 {
            return Ok(false);
        }
        if hidden == 0 || max_pos == 0 || dst_modulus == 0 {
            // `dst_modulus == 0` panics in the loop (`% 0`); keep that loud
            // path rather than inventing kernel semantics for it.
            return Ok(false);
        }
        if !all_f32(src) || dst.dtype != crate::DType::F32 {
            return Ok(false);
        }
        if self.functions.get(DFLASH_HIDDEN_SCATTER5).is_none() {
            // No fused commit ran yet in this process (seed paths,
            // non-gfx1100 ensembles): run today's loop.
            return Ok(false);
        }
        // Bounds parity: every element the kernel touches must be inside the
        // buffers, else fall back so the loop reports the violation loudly
        // instead of the kernel writing out of bounds silently.
        let Some(ring_elems) = max_pos.checked_mul(hidden) else {
            return Ok(false);
        };
        if src.iter().any(|t| t.numel() < ring_elems) {
            return Ok(false);
        }
        let Some(stride) = (num_extract as u64).checked_mul(hidden as u64) else {
            return Ok(false);
        };
        // Bound by the loop's maximum row: r ranges over r_skip..n_rows, so
        // the top row the loop can touch is dst_row_offset + n_rows - 1
        // (absolute) or dst_modulus - 1 (windowed).
        let need_rows: Option<u64> = if dst_modulus == usize::MAX {
            (dst_row_offset as u64).checked_add(n_rows as u64)
        } else {
            Some(dst_modulus as u64)
        };
        let Some(need) = need_rows.and_then(|r| r.checked_mul(stride)) else {
            return Ok(false);
        };
        if (dst.numel() as u64) < need {
            return Ok(false);
        }
        let total: u64 = (rows as u64) * 5u64 * (hidden as u64);
        debug_assert!(total <= u64::from(u32::MAX), "scatter5 grid overflow");
        let grid_x = ((total + u64::from(HIDDEN_SCATTER_BLOCK) - 1)
            / u64::from(HIDDEN_SCATTER_BLOCK)) as u32;
        let mod_u64 = dst_modulus as u64;
        if dst_modulus == usize::MAX {
            debug_assert_eq!(
                mod_u64, DST_MODULUS_ABSENT,
                "usize::MAX must map to the kernel absent-modulus sentinel"
            );
        }
        debug_assert!(start_slot <= i32::MAX as usize, "scatter5 slot overflow");
        debug_assert!(rows <= i32::MAX as usize, "scatter5 rows overflow");
        debug_assert!(r_skip <= i32::MAX as usize, "scatter5 skip overflow");
        debug_assert!(hidden <= i32::MAX as usize, "scatter5 hidden overflow");
        debug_assert!(max_pos <= i32::MAX as usize, "scatter5 max_pos overflow");
        self.bind_thread()?;
        let mut blob = hip_bridge::KernargBlob::new();
        for t in src {
            blob.push_ptr(t.buf.as_ptr());
        }
        blob.push_ptr(dst.buf.as_ptr());
        blob.push_u64(dst_row_offset as u64);
        blob.push_u64(mod_u64);
        blob.push_i32(start_slot as i32);
        blob.push_i32(rows as i32);
        blob.push_i32(r_skip as i32);
        blob.push_i32(hidden as i32);
        blob.push_i32(max_pos as i32);
        blob.pad_to(16);
        self.launch_kernel_blob(
            DFLASH_HIDDEN_SCATTER5,
            [grid_x, 1, 1],
            [HIDDEN_SCATTER_BLOCK, 1, 1],
            0,
            blob.as_mut_slice(),
        )?;
        Ok(true)
    }
}
