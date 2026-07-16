// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact load-time AoSoA4 shadow layout for HFQ4/MQ4 G256 weights.
//!
//! The on-disk layout remains untouched at offset zero.  A gfx1151-only
//! caller may append this shadow to the same immutable allocation and pass
//! [`shadow_ptr`] to a matching kernel.  Keeping both layouts in one owning
//! allocation makes the pointer stable across retained PM4 capture/replay and
//! leaves every legacy/kernel-family consumer of the original pointer intact.

use crate::GpuTensor;
use std::ffi::c_void;

pub const GROUP_WEIGHTS: usize = 256;
pub const AOS_GROUP_BYTES: usize = 136;
pub const HEADER_BYTES: usize = 8;
pub const PAYLOAD_BYTES: usize = 128;
pub const SHADOW_ALIGN: usize = 256;
pub const PAYLOAD_ALIGN: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    pub groups_per_row: usize,
    pub aos_bytes: usize,
    pub shadow_offset: usize,
    pub header_bytes: usize,
    pub payload_offset: usize,
    pub payload_bytes: usize,
    pub total_bytes: usize,
}

const fn align_up(value: usize, align: usize) -> Option<usize> {
    match value.checked_add(align - 1) {
        Some(v) => Some(v & !(align - 1)),
        None => None,
    }
}

pub fn layout(m: usize, k: usize) -> Option<Layout> {
    if m == 0 || k == 0 || !k.is_multiple_of(GROUP_WEIGHTS) {
        return None;
    }
    let groups_per_row = k / GROUP_WEIGHTS;
    if !groups_per_row.is_multiple_of(4) {
        return None;
    }
    let groups = m.checked_mul(groups_per_row)?;
    let aos_bytes = groups.checked_mul(AOS_GROUP_BYTES)?;
    let shadow_offset = align_up(aos_bytes, SHADOW_ALIGN)?;
    let header_bytes = groups.checked_mul(HEADER_BYTES)?;
    let payload_offset = align_up(header_bytes, PAYLOAD_ALIGN)?;
    let payload_bytes = groups.checked_mul(PAYLOAD_BYTES)?;
    let shadow_bytes = payload_offset.checked_add(payload_bytes)?;
    let total_bytes = shadow_offset.checked_add(shadow_bytes)?;
    Some(Layout {
        groups_per_row,
        aos_bytes,
        shadow_offset,
        header_bytes,
        payload_offset,
        payload_bytes,
        total_bytes,
    })
}

/// Preserve `aos` at offset zero and append an exact byte-reordered shadow:
/// all 8-byte scale/zero headers first, then payloads tiled as
/// `[row][group_quad][lane][group_in_quad]`.  A wave lane can consequently
/// fetch its four packed dwords with one aligned B128 operation while keeping
/// the incumbent four-accumulator arithmetic order.
pub fn append_shadow(aos: &[u8], m: usize, k: usize) -> Option<Vec<u8>> {
    let l = layout(m, k)?;
    if aos.len() != l.aos_bytes {
        return None;
    }
    let mut out = vec![0u8; l.total_bytes];
    out[..aos.len()].copy_from_slice(aos);
    let shadow = l.shadow_offset;
    let payload = shadow + l.payload_offset;
    for row in 0..m {
        for group in 0..l.groups_per_row {
            let flat_group = row * l.groups_per_row + group;
            let src = flat_group * AOS_GROUP_BYTES;
            let header_dst = shadow + flat_group * HEADER_BYTES;
            out[header_dst..header_dst + HEADER_BYTES]
                .copy_from_slice(&aos[src..src + HEADER_BYTES]);
        }
        for quad in 0..l.groups_per_row / 4 {
            for lane in 0..32 {
                for group_in_quad in 0..4 {
                    let group = quad * 4 + group_in_quad;
                    let flat_group = row * l.groups_per_row + group;
                    let src = flat_group * AOS_GROUP_BYTES + HEADER_BYTES + lane * 4;
                    let payload_dst = payload
                        + row * l.groups_per_row * PAYLOAD_BYTES
                        + quad * 4 * PAYLOAD_BYTES
                        + lane * 16
                        + group_in_quad * 4;
                    out[payload_dst..payload_dst + 4].copy_from_slice(&aos[src..src + 4]);
                }
            }
        }
    }
    Some(out)
}

/// Return the appended shadow base only when the allocation is large enough
/// to prove that it carries the exact layout for `(m, k)`.
pub fn shadow_ptr(tensor: &GpuTensor, m: usize, k: usize) -> Option<*mut c_void> {
    let l = layout(m, k)?;
    if tensor.buf.size() < l.total_bytes {
        return None;
    }
    let base = tensor.buf.as_ptr().cast::<u8>();
    Some(unsafe { base.add(l.shadow_offset).cast::<c_void>() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_headers_and_aligned_payloads_without_changing_bits() {
        let m = 2;
        let k = 1024;
        let l = layout(m, k).unwrap();
        let aos: Vec<u8> = (0..l.aos_bytes).map(|i| i as u8).collect();
        let staged = append_shadow(&aos, m, k).unwrap();
        assert_eq!(&staged[..aos.len()], aos.as_slice());
        assert_eq!(l.shadow_offset % SHADOW_ALIGN, 0);
        assert_eq!(l.payload_offset % PAYLOAD_ALIGN, 0);
        for row in 0..m {
            for group in 0..l.groups_per_row {
                let flat_group = row * l.groups_per_row + group;
                let src = flat_group * AOS_GROUP_BYTES;
                let header = l.shadow_offset + flat_group * HEADER_BYTES;
                assert_eq!(
                    &staged[header..header + HEADER_BYTES],
                    &aos[src..src + HEADER_BYTES]
                );
                for lane in 0..32 {
                    let quad = group / 4;
                    let group_in_quad = group % 4;
                    let payload = l.shadow_offset
                        + l.payload_offset
                        + row * l.groups_per_row * PAYLOAD_BYTES
                        + quad * 4 * PAYLOAD_BYTES
                        + lane * 16
                        + group_in_quad * 4;
                    assert_eq!(
                        &staged[payload..payload + 4],
                        &aos[src + HEADER_BYTES + lane * 4..src + HEADER_BYTES + lane * 4 + 4]
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_non_g256_shapes_and_wrong_source_size() {
        assert!(layout(1, 255).is_none());
        assert!(layout(1, 256).is_none());
        assert!(append_shadow(&[0; AOS_GROUP_BYTES - 1], 1, 256).is_none());
    }
}
