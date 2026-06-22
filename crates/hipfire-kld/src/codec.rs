// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-blob storage codecs for `.kldref` payloads.
//!
//! A full reference is dominated by two equal-size arrays — `top_indices` (u32)
//! and `top_log_probs` (f32) — each ~1.2 GB for a 0.8B/1175-chunk ref. The codec
//! tag travels in the ref metadata so a reader knows how to decode each blob.
//!
//! Shipped today (lossless, dependency-free):
//! - [`BlobCodec::RawU32`] / [`BlobCodec::RawF32`] — passthrough.
//! - [`BlobCodec::BitpackedIdx`] — token ids packed at `ceil(log2(n_vocab))`
//!   bits. Vocab 248,320 < 2^18 → 18 bits, a deterministic ~44% on the index
//!   array, lossless.
//!
//! Reserved (need a precision-shift validation and/or a dependency before they
//! become selectable defaults):
//! - `Fp16` — half-precision `top_log_probs` (~2×, lossy on the *reference*
//!   baseline; gate behind a measured KLD-shift tolerance).
//! - `Zstd` — general lossless wrap (needs the `zstd` dep).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum BlobCodec {
    RawU32,
    RawF32,
    /// Token ids packed at `bits` bits each (little-endian bit order).
    BitpackedIdx {
        bits: u32,
    },
    /// half-precision floats — reserved, not yet validated.
    Fp16,
    /// zstd-wrapped — reserved, needs the zstd dependency.
    Zstd {
        inner: Box<BlobCodec>,
    },
}

/// Bits needed to represent any id in `[0, n_vocab)` = `ceil(log2(n_vocab))`.
pub fn bits_for_vocab(n_vocab: usize) -> u32 {
    let m = n_vocab.max(2) as u64;
    64 - (m - 1).leading_zeros()
}

/// Pack `values` (each `< 2^bits`) into a little-endian bitstream.
pub fn bitpack(values: &[u32], bits: u32) -> Result<Vec<u8>, String> {
    if !(1..=32).contains(&bits) {
        return Err(format!("bitpack: bits must be 1..=32, got {bits}"));
    }
    let limit = if bits == 32 {
        u64::from(u32::MAX) + 1
    } else {
        1u64 << bits
    };
    let mut out = vec![0u8; (values.len() * bits as usize).div_ceil(8)];
    let mut bit_pos = 0usize;
    for &v in values {
        if (v as u64) >= limit {
            return Err(format!("bitpack: value {v} exceeds {bits}-bit range"));
        }
        for b in 0..bits as usize {
            if (v >> b) & 1 == 1 {
                out[(bit_pos + b) / 8] |= 1 << ((bit_pos + b) % 8);
            }
        }
        bit_pos += bits as usize;
    }
    Ok(out)
}

/// Unpack `count` values of `bits` bits each from a little-endian bitstream.
pub fn bitunpack(buf: &[u8], count: usize, bits: u32) -> Result<Vec<u32>, String> {
    if !(1..=32).contains(&bits) {
        return Err(format!("bitunpack: bits must be 1..=32, got {bits}"));
    }
    let need = (count * bits as usize).div_ceil(8);
    if buf.len() < need {
        return Err(format!("bitunpack: buffer {} < needed {need}", buf.len()));
    }
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut v = 0u32;
        for b in 0..bits as usize {
            let byte = buf[(bit_pos + b) / 8];
            if (byte >> ((bit_pos + b) % 8)) & 1 == 1 {
                v |= 1 << b;
            }
        }
        out.push(v);
        bit_pos += bits as usize;
    }
    Ok(out)
}

/// Saving ratio of bit-packing `n_vocab` ids vs raw u32 (for logging/metadata).
pub fn bitpack_ratio(n_vocab: usize) -> f64 {
    bits_for_vocab(n_vocab) as f64 / 32.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_for_vocab_is_ceil_log2() {
        assert_eq!(bits_for_vocab(248_320), 18); // 2^17=131072 < 248320 <= 2^18
        assert_eq!(bits_for_vocab(256), 8);
        assert_eq!(bits_for_vocab(257), 9);
        assert_eq!(bits_for_vocab(65536), 16);
    }

    #[test]
    fn bitpack_round_trips_at_vocab_width() {
        let bits = bits_for_vocab(248_320); // 18
        let vals: Vec<u32> = (0..1000).map(|i| (i * 241) % 248_320).collect();
        let packed = bitpack(&vals, bits).unwrap();
        // 18 bits × 1000 = 18000 bits = 2250 bytes (vs 4000 raw)
        assert_eq!(packed.len(), (1000 * 18usize).div_ceil(8));
        let back = bitunpack(&packed, vals.len(), bits).unwrap();
        assert_eq!(back, vals);
    }

    #[test]
    fn bitpack_rejects_out_of_range() {
        assert!(bitpack(&[8u32], 3).is_err()); // 8 needs 4 bits
        assert!(bitpack(&[7u32], 3).is_ok());
    }

    #[test]
    fn bitpack_edge_widths() {
        for bits in [1u32, 7, 8, 17, 18, 31, 32] {
            let limit = if bits == 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            let vals = vec![0u32, 1, limit, limit / 2];
            let packed = bitpack(&vals, bits).unwrap();
            assert_eq!(bitunpack(&packed, vals.len(), bits).unwrap(), vals);
        }
    }

    #[test]
    fn ratio_is_under_half_for_real_vocab() {
        assert!(bitpack_ratio(248_320) < 0.57); // 18/32 = 0.5625
    }
}
