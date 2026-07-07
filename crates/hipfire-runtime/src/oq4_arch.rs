// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OQ4 arch-packing: the single source of truth for converting canonical
//! portable OQ4 weights into the arch-optimal combined device layout.
//!
//! Canonical on-disk OQ4 (`quant_type` [`OQ4_CANONICAL_QT`], `Oq4G256`) stores
//! each 256-group as `[f16 scale][128 nibbles]`, row-contiguous — the portable,
//! arch-independent form that ships everywhere. The GPU loaders consume an
//! arch *combined* layout (split nibbles + split f32 scales + interleaved decode
//! records); [`oq4_pack_arch_combined`] performs that transform. The
//! `hipfire optimize`/`oq4_repack` tool applies it ahead of load and stamps the
//! result as [`OQ4_ARCH_PACKED_QT`] (`Oq4G256ArchPacked`), which loaders then
//! upload verbatim.
//!
//! Every model loader and the offline tool route through this module so the
//! transform, its length contract, the quant-type codes, and the
//! canonical-vs-packed load decision cannot drift across crates.

use std::borrow::Cow;

use hipfire_rdna::DType;

use crate::quant::{f16_to_f32, QuantType};

/// Canonical portable on-disk OQ4 quant-type code (`Oq4G256`, repacked at load).
pub const OQ4_CANONICAL_QT: u8 = QuantType::Oq4G256.code();
/// Arch-packed OQ4 quant-type code (`Oq4G256ArchPacked`, uploaded verbatim).
pub const OQ4_ARCH_PACKED_QT: u8 = QuantType::Oq4G256ArchPacked.code();

/// Byte length of the OQ4 arch combined device layout for an `[m, k]` matrix:
/// `[split nibbles m*(k/2)] [split f32 scales m*ng] [interleaved m*ng*132]`.
pub fn oq4_arch_combined_len(m: usize, k: usize) -> usize {
    let ng = k / 256;
    m * (k / 2) + m * ng * 4 + m * ng * (4 + 128)
}

/// Repack canonical on-disk OQ4 (`OQ4_CANONICAL_QT`: `[f16 scale][128 nibbles]`
/// per 256-group, row-contiguous) into the arch combined device layout uploaded
/// by the loader. SINGLE source of truth for that transform — every qt=34 load
/// path and the offline optimize tool call it, so they cannot drift.
///
/// Output layout (`[m, k]`, `ng = k/256`):
///   `[split nibbles m*(k/2)]` — prefill MMQ/f16 (`sub_offset 0`)
///   `[split f32 scales m*ng]` — prefill weight-scale region (`sub_offset m*(k/2)`)
///   `[interleaved m*ng*132]`  — decode GEMVs: per group `[f32 scale][128 nibbles]`
///                               contiguous → one coalesced stream (mq4-style).
pub fn oq4_pack_arch_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    const BLOCK: usize = 130; // 2 (f16 scale) + 128 nibbles
    const ILB: usize = 4 + 128; // [f32 scale][128 nibbles]
    assert_eq!(k % GROUP, 0, "OQ4G256 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let packed_bytes = m * (k / 2);
    let scales_bytes = m * ng * 4;
    let il_bytes = m * ng * ILB;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OQ4G256 weight byte length {} != M*ng*130 = {expect} (M={m} K={k})",
        data.len()
    );
    let mut combined = vec![0u8; packed_bytes + scales_bytes + il_bytes];
    let il_base = packed_bytes + scales_bytes;
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let dst = r * (k / 2) + g * (GROUP / 2);
            combined[dst..dst + 128].copy_from_slice(&data[src + 2..src + BLOCK]);
            let so = packed_bytes + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
            let io = il_base + (r * ng + g) * ILB;
            combined[io..io + 4].copy_from_slice(&scale.to_le_bytes());
            combined[io + 4..io + ILB].copy_from_slice(&data[src + 2..src + BLOCK]);
        }
    }
    combined
}

/// Resolve an OQ4 weight blob to the exact bytes + `DType` a loader should
/// upload. Returns `None` for quant-types outside the OQ4 canonical/arch-packed
/// pair so callers keep their own arms for OQ+/OQ8/MQ/F16/… .
///
/// - [`OQ4_CANONICAL_QT`] (34): repacks into the arch combined layout (owned).
/// - [`OQ4_ARCH_PACKED_QT`] (37): the on-disk data already IS the combined
///   layout — returned borrowed (zero-copy) after a length check.
///
/// Panics on a corrupt qt=37 length, matching [`oq4_pack_arch_combined`]'s
/// assert-on-malformed contract (a stale/garbage layout is refused, not read as
/// garbage). The `Cow` keeps the fast arch-packed load path allocation-free.
pub fn oq4_arch_load(qt: u8, data: &[u8], m: usize, k: usize) -> Option<(Cow<'_, [u8]>, DType)> {
    match qt {
        OQ4_CANONICAL_QT => Some((
            Cow::Owned(oq4_pack_arch_combined(data, m, k)),
            DType::Oq4G256,
        )),
        OQ4_ARCH_PACKED_QT => {
            let expect = oq4_arch_combined_len(m, k);
            assert_eq!(
                data.len(),
                expect,
                "Oq4G256ArchPacked byte length {} != arch combined len {expect} (M={m} K={k}); \
                 stale or corrupt arch-packed artifact",
                data.len()
            );
            Some((Cow::Borrowed(data), DType::Oq4G256))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 256-group canonical block: `[f16 scale (2)][128 nibbles]`.
    fn canonical_block(scale_f16_bits: u16, nibbles: &[u8; 128]) -> Vec<u8> {
        let mut v = Vec::with_capacity(130);
        v.extend_from_slice(&scale_f16_bits.to_le_bytes());
        v.extend_from_slice(nibbles);
        v
    }

    #[test]
    fn pack_matches_golden_layout() {
        // scale = 1.0 (f16 bits 0x3C00 -> f32 1.0), nibbles 0..128.
        let nibbles: [u8; 128] = std::array::from_fn(|i| i as u8);
        let data = canonical_block(0x3C00, &nibbles);
        let (m, k) = (1, 256);
        let out = oq4_pack_arch_combined(&data, m, k);

        assert_eq!(out.len(), oq4_arch_combined_len(m, k));
        assert_eq!(out.len(), 264); // 128 + 4 + 132
        let one = 1.0f32.to_le_bytes();
        assert_eq!(&out[0..128], &nibbles[..]); // split nibbles
        assert_eq!(&out[128..132], &one); // split f32 scale
        assert_eq!(&out[132..136], &one); // interleaved f32 scale
        assert_eq!(&out[136..264], &nibbles[..]); // interleaved nibbles
    }

    #[test]
    fn arch_load_packs_canonical_borrows_packed_and_skips_others() {
        let nibbles: [u8; 128] = std::array::from_fn(|i| (i as u8).wrapping_mul(3));
        let data = canonical_block(0x3C00, &nibbles);
        let (m, k) = (1, 256);

        // 34: owned, equal to the direct transform.
        let (canon, dt) = oq4_arch_load(OQ4_CANONICAL_QT, &data, m, k).unwrap();
        assert_eq!(dt, DType::Oq4G256);
        assert!(matches!(canon, Cow::Owned(_)));
        assert_eq!(
            canon.as_ref(),
            oq4_pack_arch_combined(&data, m, k).as_slice()
        );

        // 37: borrowed (zero-copy), returned verbatim.
        let packed = vec![7u8; oq4_arch_combined_len(m, k)];
        let (verbatim, dt) = oq4_arch_load(OQ4_ARCH_PACKED_QT, &packed, m, k).unwrap();
        assert_eq!(dt, DType::Oq4G256);
        assert!(matches!(verbatim, Cow::Borrowed(_)));
        assert_eq!(verbatim.as_ref(), packed.as_slice());

        // Non-OQ4 code: None (caller handles other formats).
        assert!(oq4_arch_load(99, &data, m, k).is_none());
    }
}
