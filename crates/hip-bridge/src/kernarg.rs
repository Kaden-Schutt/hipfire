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
        Self { buf: Vec::with_capacity(64) }
    }

    /// Construct with a pre-reserved capacity — avoids a realloc when the
    /// final size is known.
    pub fn with_capacity(cap: usize) -> Self {
        Self { buf: Vec::with_capacity(cap) }
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
}
