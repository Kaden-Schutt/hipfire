// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Kernarg blob builder for `HipRuntime::launch_kernel_blob`.
//!
//! The blob is a contiguous byte buffer laid out according to the kernel's
//! C ABI: each field is placed at its natural alignment, padding is inserted
//! where needed, and the total length matches the kernel's expected kernarg
//! struct size.
//!
//! Example — the `mul_f32` kernel:
//!
//! ```ignore
//! extern "C" __global__ void mul_f32(
//!     const float* a, const float* b, float* c, int n
//! );
//! ```
//!
//! Corresponding blob:
//!
//! ```ignore
//! let mut k = KernargBlob::new();
//! k.push_ptr(a.as_ptr());
//! k.push_ptr(b.as_ptr());
//! k.push_ptr(c.as_ptr());
//! k.push_i32(n);
//! // k.as_bytes() is 28 bytes: [8 | 8 | 8 | 4]
//! gpu.launch_kernel_blob(func, grid, block, 0, stream, k.as_mut_slice())?;
//! ```
//!
//! For the graph-capture flow the caller typically keeps the `KernargBlob`
//! alive for the lifetime of the executable graph (via a Vec<KernargBlob>
//! inside the graph owner), since HIP graph capture on gfx1100/ROCm 6.3 only
//! records the *pointer* to the blob — the blob itself must not move or be
//! freed until the graph is destroyed. For one-shot launches the blob can be
//! stack-local and dropped immediately after `launch_kernel_blob` returns.

use std::ffi::c_void;

/// One kernel argument, tagged by its ABI kind.
///
/// A single `KernArg` list is the single source of truth for both launch
/// ABIs: the packed [`KernargBlob`] (graph-capture path) is built with
/// [`KernargBlob::push_arg`], and the `kernelParams` pointer array is derived
/// with [`KernArg::param_ptr`]. Because both come from the same list, the two
/// representations cannot silently disagree — the dual-maintenance hazard the
/// old hand-written blob-closure + `Vec<*mut c_void>` pairs carried at every
/// launch site (review 2026-07-03 §3.1).
#[derive(Clone, Copy, Debug)]
pub enum KernArg {
    Ptr(*const c_void),
    I32(i32),
    U32(u32),
    F32(f32),
    U64(u64),
}

impl KernArg {
    /// Address of this argument's payload, for the `kernelParams` array HIP's
    /// non-capture launch path expects (a pointer to each argument value).
    ///
    /// The returned pointer borrows `self`, so it is valid only while the
    /// `KernArg` (typically an element of a caller-owned `&[KernArg]` slice)
    /// is alive. HIP copies the pointed-to bytes during the launch call, so
    /// passing `&kernargs![...]` as a temporary that lives across the launch
    /// statement is sound — the same lifetime guarantee the old per-site
    /// `Vec<*mut c_void>` of addresses-of-locals relied on.
    pub fn param_ptr(&self) -> *mut c_void {
        match self {
            KernArg::Ptr(p) => p as *const *const c_void as *mut c_void,
            KernArg::I32(v) => v as *const i32 as *mut c_void,
            KernArg::U32(v) => v as *const u32 as *mut c_void,
            KernArg::F32(v) => v as *const f32 as *mut c_void,
            KernArg::U64(v) => v as *const u64 as *mut c_void,
        }
    }
}

impl From<*const c_void> for KernArg {
    fn from(p: *const c_void) -> Self {
        KernArg::Ptr(p)
    }
}
impl From<*mut c_void> for KernArg {
    fn from(p: *mut c_void) -> Self {
        KernArg::Ptr(p as *const c_void)
    }
}
impl From<i32> for KernArg {
    fn from(v: i32) -> Self {
        KernArg::I32(v)
    }
}
impl From<u32> for KernArg {
    fn from(v: u32) -> Self {
        KernArg::U32(v)
    }
}
impl From<f32> for KernArg {
    fn from(v: f32) -> Self {
        KernArg::F32(v)
    }
}
impl From<u64> for KernArg {
    fn from(v: u64) -> Self {
        KernArg::U64(v)
    }
}

/// A growable kernarg byte buffer with natural-alignment padding semantics.
///
/// Fields are appended with `push_ptr`, `push_u32`, `push_i32`, `push_f32`;
/// each push pads to the field's natural alignment before writing its bytes.
/// Final buffer may need a tail pad to the kernel's total alignment — HIP's
/// kernarg loader on gfx1100 accepts the unpadded tail fine in practice, but
/// you can call `pad_to(16)` before launching for safety on unknown archs.
pub struct KernargBlob {
    buf: Vec<u8>,
}

impl KernargBlob {
    /// Construct an empty blob.
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(64),
        }
    }

    /// Construct with a pre-reserved capacity — avoids a realloc when the
    /// final size is known.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    /// Current offset in bytes (useful for debugging alignment bugs).
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Pad the buffer with zero bytes until its length is a multiple of `align`.
    #[inline]
    fn align_to(&mut self, align: usize) {
        debug_assert!(align.is_power_of_two(), "alignment must be power of two");
        let cur = self.buf.len();
        let misaligned = cur & (align - 1);
        if misaligned != 0 {
            self.buf.resize(cur + (align - misaligned), 0);
        }
    }

    /// Append an 8-byte pointer, padded to 8-byte alignment.
    pub fn push_ptr(&mut self, ptr: *const c_void) {
        self.align_to(8);
        let bytes = (ptr as usize).to_ne_bytes();
        self.buf.extend_from_slice(&bytes);
    }

    /// Append a 4-byte unsigned int, padded to 4-byte alignment.
    pub fn push_u32(&mut self, v: u32) {
        self.align_to(4);
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }

    /// Append a 4-byte signed int, padded to 4-byte alignment.
    pub fn push_i32(&mut self, v: i32) {
        self.align_to(4);
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }

    /// Append a 4-byte float, padded to 4-byte alignment.
    pub fn push_f32(&mut self, v: f32) {
        self.align_to(4);
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }

    /// Append an 8-byte unsigned long long, padded to 8-byte alignment.
    pub fn push_u64(&mut self, v: u64) {
        self.align_to(8);
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }

    /// Append one typed argument, dispatching to the right `push_*` by kind.
    /// This is the blob-side counterpart of [`KernArg::param_ptr`]; feeding a
    /// `&[KernArg]` through both keeps the packed and pointer ABIs in lockstep.
    pub fn push_arg(&mut self, arg: &KernArg) {
        match *arg {
            KernArg::Ptr(p) => self.push_ptr(p),
            KernArg::I32(v) => self.push_i32(v),
            KernArg::U32(v) => self.push_u32(v),
            KernArg::F32(v) => self.push_f32(v),
            KernArg::U64(v) => self.push_u64(v),
        }
    }

    /// Pad the buffer to a multiple of `align` bytes. Call before launch if
    /// the arch's loader is picky about tail padding; typically unnecessary
    /// on gfx1100 / ROCm 6.x.
    pub fn pad_to(&mut self, align: usize) {
        self.align_to(align);
    }

    /// Borrow the underlying byte buffer as a mutable slice suitable for
    /// passing to `HipRuntime::launch_kernel_blob`.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Borrow the underlying byte buffer as an immutable slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Consume and return the raw Vec — useful when storing captured kernargs
    /// in a graph-owned arena.
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

impl Default for KernargBlob {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_ptr_then_i32_aligns_correctly() {
        let mut k = KernargBlob::new();
        k.push_ptr(0x1000 as *const c_void);
        k.push_ptr(0x2000 as *const c_void);
        k.push_i32(42);
        // 8 + 8 + 4 = 20 bytes, no padding between because ptr→i32 is naturally
        // aligned (ptr is 8, len is 16, i32 needs align 4 which is already
        // satisfied, so no pad, ends at 20).
        assert_eq!(k.len(), 20);
    }

    #[test]
    fn push_i32_then_ptr_pads_between() {
        let mut k = KernargBlob::new();
        k.push_i32(42);
        k.push_ptr(0x1000 as *const c_void);
        // 4 bytes + 4 bytes pad + 8 bytes ptr = 16.
        assert_eq!(k.len(), 16);
    }

    #[test]
    fn pad_to_16() {
        let mut k = KernargBlob::new();
        k.push_i32(42);
        k.pad_to(16);
        assert_eq!(k.len(), 16);
    }

    #[test]
    fn push_arg_matches_manual_push_sequence() {
        // The rmsnorm arg list: three pointers + i32 + f32.
        let args = [
            KernArg::Ptr(0x1000 as *const c_void),
            KernArg::Ptr(0x2000 as *const c_void),
            KernArg::Ptr(0x3000 as *const c_void),
            KernArg::I32(256),
            KernArg::F32(1e-6),
        ];
        let mut via_args = KernargBlob::new();
        for a in &args {
            via_args.push_arg(a);
        }
        let mut manual = KernargBlob::new();
        manual.push_ptr(0x1000 as *const c_void);
        manual.push_ptr(0x2000 as *const c_void);
        manual.push_ptr(0x3000 as *const c_void);
        manual.push_i32(256);
        manual.push_f32(1e-6);
        assert_eq!(via_args.as_bytes(), manual.as_bytes());
        assert_eq!(via_args.len(), 32); // 8+8+8 ptrs + 4 i32 + 4 f32
    }

    #[test]
    fn param_ptr_reads_back_the_payload() {
        // A pointer arg's param_ptr points at the stored pointer value.
        let arg = KernArg::Ptr(0xdead_beef as *const c_void);
        let pp = arg.param_ptr() as *const *const c_void;
        assert_eq!(unsafe { *pp }, 0xdead_beef as *const c_void);
        // A scalar arg's param_ptr points at the stored scalar.
        let arg = KernArg::I32(-7);
        assert_eq!(unsafe { *(arg.param_ptr() as *const i32) }, -7);
        let arg = KernArg::F32(2.5);
        assert_eq!(unsafe { *(arg.param_ptr() as *const f32) }, 2.5);
    }

    #[test]
    fn push_arg_alignment_pads_like_typed_pushes() {
        // i32 then ptr must insert a 4-byte pad before the 8-aligned pointer.
        let args = [KernArg::I32(1), KernArg::Ptr(0x10 as *const c_void)];
        let mut b = KernargBlob::new();
        for a in &args {
            b.push_arg(a);
        }
        assert_eq!(b.len(), 16);
    }
}
