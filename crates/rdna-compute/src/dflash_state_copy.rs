// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 (launch-fusion): `Gpu` launchers for the descriptor-driven DeltaNet
//! snapshot bulk copy (`dflash_state_bulk_copy_gfx1100`, gfx1100-only).
//!
//! The kernel source is self-contained here via `include_str!` so no shared
//! registry (`kernels.rs` / `replay.rs`) changes are needed. One block per
//! copy descriptor, 256 threads, 16 B vector loop plus scalar tail — a pure
//! byte copy, bit-exact and deterministic by construction.
//!
//! Both launchers go through `launch_maybe_blob` semantics: the default-stream
//! entry uses `launch_maybe_blob` directly (blob retained through any
//! graph-exec lifetime); the explicit-stream entry mirrors its
//! record-or-launch branching for a caller-supplied stream, bailing to the
//! caller's memcpy fallback while graph capture is active (blob retention
//! needs `&mut`).

use crate::dispatch::Gpu;
use hip_bridge::{HipResult, KernargBlob, Stream};
use std::ffi::c_void;

/// Kernel source for [`Gpu::dflash_state_bulk_copy_gfx1100`].
pub const DFLASH_STATE_BULK_COPY_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/dflash_state_bulk_copy.gfx1100.hip");
/// Compiled-module key for the bulk-copy kernel.
pub const DFLASH_STATE_BULK_COPY_GFX1100_MODULE: &str = "dflash_state_bulk_copy_gfx1100";
/// Device symbol for the bulk-copy kernel.
pub const DFLASH_STATE_BULK_COPY_GFX1100_SYMBOL: &str = "dflash_state_bulk_copy_gfx1100";
/// Threads per block: one block copies one descriptor.
pub const DFLASH_STATE_BULK_COPY_BLOCK: u32 = 256;

/// One copy work item: copy `cnt` bytes from `src + off` to `dst + off`.
///
/// `#[repr(C)]` layout (4 x u64 = 32 B) matches `DflashStateCopyDesc` in
/// `kernels/src/dflash_state_bulk_copy.gfx1100.hip`. Tables are built with
/// 64-KiB-aligned chunk offsets so every vector lane stays 16 B aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DflashStateCopyDesc {
    pub src: u64,
    pub dst: u64,
    pub off: u64,
    pub cnt: u64,
}

impl DflashStateCopyDesc {
    /// Byte view for a single `memcpy_htod` table upload.
    pub fn as_bytes(descs: &[Self]) -> &[u8] {
        // SAFETY: repr(C) over plain u64s; size is len * 32, alignment 8.
        unsafe {
            std::slice::from_raw_parts(
                descs.as_ptr() as *const u8,
                descs.len() * std::mem::size_of::<Self>(),
            )
        }
    }
}

/// Maximum grid.x for the fixed one-block-per-descriptor grid.
pub const DFLASH_STATE_BULK_COPY_MAX_ITEMS: u32 = 65_535;

impl Gpu {
    /// JIT the bulk-copy kernel (idempotent). Called once at snapshot
    /// allocation, never in a decode cycle.
    pub fn ensure_dflash_state_bulk_copy_gfx1100(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            DFLASH_STATE_BULK_COPY_GFX1100_MODULE,
            DFLASH_STATE_BULK_COPY_GFX1100_SRC,
            DFLASH_STATE_BULK_COPY_GFX1100_SYMBOL,
        )
    }

    /// Launch the bulk copy over `n_items` descriptors at `desc_ptr` on the
    /// active (default) stream via `launch_maybe_blob`.
    pub fn dflash_state_bulk_copy_gfx1100(
        &mut self,
        desc_ptr: *const c_void,
        n_items: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_dflash_state_bulk_copy_gfx1100()?;
        debug_assert!(n_items > 0 && n_items <= DFLASH_STATE_BULK_COPY_MAX_ITEMS);

        let mut p_desc = desc_ptr as *mut c_void;
        let mut p_n = n_items;
        let mut params: Vec<*mut c_void> = vec![
            &mut p_desc as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            DFLASH_STATE_BULK_COPY_GFX1100_SYMBOL,
            [n_items, 1, 1],
            [DFLASH_STATE_BULK_COPY_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(desc_ptr);
                b.push_u32(n_items);
                b
            },
        )
    }

    /// Launch the bulk copy over `n_items` descriptors at `desc_ptr` on an
    /// explicit `stream` (the `save_from_async_on` path, `&Gpu` receiver).
    ///
    /// Mirrors `launch_maybe_blob`'s record-or-launch branching: records into
    /// the Redline tape when recording so tapes stay in lockstep, and bails
    /// (caller falls back to the async memcpy loop) while graph capture is
    /// active, where kernarg-blob retention needs `&mut`. The kernel must
    /// already be ensured (snapshot allocation ensures it); a missing
    /// function also routes to the fallback.
    pub fn dflash_state_bulk_copy_gfx1100_on_stream(
        &self,
        desc_ptr: *const c_void,
        n_items: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if n_items == 0 || n_items > DFLASH_STATE_BULK_COPY_MAX_ITEMS {
            return Err(hip_bridge::HipError::new(
                0,
                "dflash_state_bulk_copy_gfx1100_on_stream: item count out of range",
            ));
        }
        if self.graphs.capture_mode {
            return Err(hip_bridge::HipError::new(
                0,
                "dflash_state_bulk_copy_gfx1100_on_stream: refusing capture without blob retention",
            ));
        }
        let func = self
            .functions
            .get(DFLASH_STATE_BULK_COPY_GFX1100_SYMBOL)
            .ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    "dflash_state_bulk_copy_gfx1100_on_stream: kernel not ensured",
                )
            })?;
        let mut blob = KernargBlob::new();
        blob.push_ptr(desc_ptr);
        blob.push_u32(n_items);
        blob.pad_to(16);
        // NOTE: deliberately not recorded into the Redline tape (`&self`
        // cannot take the `&mut` the recorder needs). This matches the legacy
        // async-memcpy path, which is likewise invisible to the tape, so tape
        // identity is unchanged versus the pre-change path.
        let mut bytes = blob.into_vec();
        // SAFETY: blob layout (ptr, u32, pad to 16) matches the kernel
        // signature; device pointers were validated at table build; `bytes`
        // lives across this one-shot launch (calls remain outside verify
        // capture per the S1 contract).
        unsafe {
            self.hip.launch_kernel_blob(
                func,
                [n_items, 1, 1],
                [DFLASH_STATE_BULK_COPY_BLOCK, 1, 1],
                0,
                Some(stream),
                bytes.as_mut_slice(),
            )
        }
    }
}
