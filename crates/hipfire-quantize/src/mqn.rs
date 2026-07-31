// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ*N-G256 — candidate successors to the MQ4/MQ3/MQ2 G256 family.
//! Encoders plus CPU dequant references.
//!
//! Each width keeps its current byte count, its FWHT-256 rotation, and its
//! existing bit packing. Only the 8-byte header changes:
//!
//! | | header | grid |
//! |---|---|---|
//! | MQ*G256 | `[f32 scale][f32 min]` | uniform, min/max fit |
//! | MQ*N    | `[f32 master][4 x u8 E4M3]` | Lloyd-Max Gaussian codebook |
//!
//! After a 256-point FWHT each coordinate is a sum of 256 sign-flipped weights,
//! so by the CLT it is near-Gaussian. That yields three things at every width:
//! the distribution is symmetric (so the f32 zero-point is nearly dead weight
//! and can be spent on four E4M3 sub-scales, one per 64 weights); the optimal
//! reconstruction levels are a mathematical constant rather than calibration
//! data; and the min/max fit is measurably far from optimal.
//!
//! **The min/max penalty grows sharply as bits shrink** — that is the headline
//! for the sub-4-bit formats. Ideal-fit headroom over min/max on a Gaussian:
//!
//! | levels | bits | headroom |
//! |---|---|---|
//! | 16 | 4 | +2.18 dB |
//! | 8  | 3 | +3.19 dB |
//! | 4  | 2 | **+5.23 dB** |
//!
//! Byte budgets are unchanged from the uniform formats, so nothing here costs
//! bandwidth:
//!
//! | spec | header | payload | total | bpw |
//! |---|---|---|---|---|
//! | [`MQ4N`] | 8 | 128 | 136 | 4.25 |
//! | [`MQ3N`] | 8 | 96 | 104 | 3.25 |
//! | [`MQ2N`] | 8 | 64 | 72 | 2.25 |
//!
//! NOTE: no `QuantType` discriminants are assigned and no GPU kernel consumes
//! these. Validated encoders plus reference decoders, not shippable formats.

use crate::fp8::{e4m3_decode, e4m3_encode_roundup};

/// Weights per rotation/scale group.
pub const GROUP: usize = 256;
/// Weights sharing one E4M3 sub-scale.
pub const SUB: usize = 64;
/// Sub-scales per group.
pub const N_SUB: usize = GROUP / SUB;
/// Header bytes: `[f32 master][4 x u8 E4M3]`.
pub const HEADER: usize = 8;

/// Lloyd-Max optimal levels for a unit Gaussian, normalized so `max|c| == 1`.
///
/// Solved on the Gaussian density itself, then symmetrized, so these are exact
/// constants rather than a fit to any model. Each verified against published
/// Lloyd-Max distortion to within 0.3%.
pub const CODEBOOK_4BIT: [f32; 16] = [
    -1.00000000,
    -0.75716355,
    -0.59212931,
    -0.45972187,
    -0.34485254,
    -0.24034313,
    -0.14200753,
    -0.04698658,
    0.04698658,
    0.14200753,
    0.24034313,
    0.34485254,
    0.45972187,
    0.59212931,
    0.75716355,
    1.00000000,
];

/// See [`CODEBOOK_4BIT`]. Verified D=0.034452 vs published 0.034545.
pub const CODEBOOK_3BIT: [f32; 8] = [
    -1.00000000,
    -0.62450892,
    -0.35131243,
    -0.11389422,
    0.11389422,
    0.35131243,
    0.62450892,
    1.00000000,
];

/// See [`CODEBOOK_4BIT`]. Verified D=0.117245 vs published 0.117493.
pub const CODEBOOK_2BIT: [f32; 4] = [-1.00000000, -0.29977143, 0.29977143, 1.00000000];

/// One member of the MQ*N family.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// Bits per weight in the payload.
    pub bits: u8,
    /// Encoded bytes per 256-weight group, header included.
    pub group_bytes: usize,
    /// Reconstruction levels, ascending.
    pub codebook: &'static [f32],
}

/// 4-bit, 136 B/group — same size as MQ4G256.
pub const MQ4N: Spec = Spec {
    bits: 4,
    group_bytes: 136,
    codebook: &CODEBOOK_4BIT,
};
/// 3-bit, 104 B/group — same size as uniform MQ3G256, and 8 B *smaller* than
/// the shipped MQ3G256Lloyd (112 B) which it also outperforms.
pub const MQ3N: Spec = Spec {
    bits: 3,
    group_bytes: 104,
    codebook: &CODEBOOK_3BIT,
};
/// 2-bit, 72 B/group — same size as both MQ2G256 and MQ2G256Lloyd.
pub const MQ2N: Spec = Spec {
    bits: 2,
    group_bytes: 72,
    codebook: &CODEBOOK_2BIT,
};

/// Sub-scale search grid, as a fraction of the sub-block maximum. Values above
/// 1.0 are included because the outermost level sits at ±1: overshooting
/// slightly can beat clipping when a sub-block has one dominant outlier.
const ALPHA_LO: f32 = 0.50;
const ALPHA_HI: f32 = 1.15;
const ALPHA_STEPS: usize = 20;

impl Spec {
    /// Payload bytes, excluding the header.
    pub const fn payload_bytes(&self) -> usize {
        GROUP * self.bits as usize / 8
    }

    /// Index of the nearest codebook entry.
    ///
    /// Exact midpoints round UP. This affects encoding only — a decoder just
    /// reads the emitted index — but it must stay fixed, since flipping it
    /// would change emitted bytes for values landing on a boundary. `v == 0.0`
    /// is the common case and resolves upward.
    #[inline]
    pub fn nearest_code(&self, v: f32) -> u8 {
        let cb = self.codebook;
        let mut idx = 0u8;
        for i in 1..cb.len() {
            if v >= (cb[i - 1] + cb[i]) * 0.5 {
                idx = i as u8;
            } else {
                break;
            }
        }
        idx
    }

    #[inline]
    fn sub_sq_err(&self, vals: &[f32], eff: f32) -> f32 {
        if !(eff > 0.0) {
            return f32::INFINITY;
        }
        let inv = 1.0 / eff;
        let mut e = 0.0f32;
        for &v in vals {
            let d = v - self.codebook[self.nearest_code(v * inv) as usize] * eff;
            e += d * d;
        }
        e
    }

    /// Pack 256 codes into the payload using the SAME bit layout the existing
    /// uniform kernels already unpack, so kernel-side unpack logic transfers:
    /// 4-bit two per byte low-nibble-first; 3-bit as 32 chunks of 8 codes in
    /// 3 bytes (cross-byte, matching `quantize_mq3g256`); 2-bit four per byte.
    fn pack(&self, codes: &[u8; GROUP], out: &mut [u8]) {
        match self.bits {
            4 => {
                for i in 0..GROUP / 2 {
                    out[i] = codes[2 * i] | (codes[2 * i + 1] << 4);
                }
            }
            3 => {
                for chunk in 0..32 {
                    let q = &codes[chunk * 8..chunk * 8 + 8];
                    let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
                    let b1 = ((q[2] >> 2) & 1)
                        | ((q[3] & 7) << 1)
                        | ((q[4] & 7) << 4)
                        | ((q[5] & 1) << 7);
                    let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);
                    out[chunk * 3] = b0;
                    out[chunk * 3 + 1] = b1;
                    out[chunk * 3 + 2] = b2;
                }
            }
            2 => {
                for i in 0..64 {
                    let mut b = 0u8;
                    for j in 0..4 {
                        b |= (codes[4 * i + j] & 3) << (j * 2);
                    }
                    out[i] = b;
                }
            }
            b => unreachable!("unsupported width {b}"),
        }
    }

    /// Inverse of [`Spec::pack`].
    fn unpack(&self, payload: &[u8], codes: &mut [u8; GROUP]) {
        match self.bits {
            4 => {
                for i in 0..GROUP / 2 {
                    codes[2 * i] = payload[i] & 0xF;
                    codes[2 * i + 1] = payload[i] >> 4;
                }
            }
            3 => {
                for chunk in 0..32 {
                    let (b0, b1, b2) = (
                        payload[chunk * 3],
                        payload[chunk * 3 + 1],
                        payload[chunk * 3 + 2],
                    );
                    let q = &mut codes[chunk * 8..chunk * 8 + 8];
                    q[0] = b0 & 7;
                    q[1] = (b0 >> 3) & 7;
                    q[2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2);
                    q[3] = (b1 >> 1) & 7;
                    q[4] = (b1 >> 4) & 7;
                    q[5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1);
                    q[6] = (b2 >> 2) & 7;
                    q[7] = (b2 >> 5) & 7;
                }
            }
            2 => {
                for i in 0..64 {
                    for j in 0..4 {
                        codes[4 * i + j] = (payload[i] >> (j * 2)) & 3;
                    }
                }
            }
            b => unreachable!("unsupported width {b}"),
        }
    }

    /// Encode one already-rotated 256-weight group. `out` must be
    /// [`Spec::group_bytes`] long.
    pub fn encode_group(&self, group: &[f32; GROUP], out: &mut [u8]) {
        debug_assert_eq!(out.len(), self.group_bytes);
        let master = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        out[0..4].copy_from_slice(&master.to_le_bytes());
        if !(master > 0.0) {
            out[4..].fill(0);
            return;
        }
        let inv_master = 1.0 / master;

        let mut eff = [0.0f32; N_SUB];
        for s in 0..N_SUB {
            let vals = &group[s * SUB..(s + 1) * SUB];
            let submax = vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let (mut best_byte, mut best_err) = (0u8, f32::INFINITY);
            for a in 0..ALPHA_STEPS {
                let alpha =
                    ALPHA_LO + (ALPHA_HI - ALPHA_LO) * (a as f32) / (ALPHA_STEPS - 1) as f32;
                let byte = e4m3_encode_roundup(submax * alpha * inv_master);
                let err = self.sub_sq_err(vals, master * e4m3_decode(byte));
                if err < best_err {
                    best_err = err;
                    best_byte = byte;
                }
            }
            out[4 + s] = best_byte;
            eff[s] = master * e4m3_decode(best_byte);
        }

        let mut codes = [0u8; GROUP];
        for i in 0..GROUP {
            let e = eff[i / SUB];
            codes[i] = if e > 0.0 { self.nearest_code(group[i] / e) } else { 0 };
        }
        self.pack(&codes, &mut out[HEADER..]);
    }

    /// Reference decode of one group back to the rotated domain. Any GPU kernel
    /// is required to agree with this bit-for-bit.
    pub fn decode_group(&self, blk: &[u8], out: &mut [f32; GROUP]) {
        debug_assert_eq!(blk.len(), self.group_bytes);
        let master = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        let mut eff = [0.0f32; N_SUB];
        for s in 0..N_SUB {
            eff[s] = master * e4m3_decode(blk[4 + s]);
        }
        let mut codes = [0u8; GROUP];
        self.unpack(&blk[HEADER..], &mut codes);
        for i in 0..GROUP {
            out[i] = self.codebook[codes[i] as usize] * eff[i / SUB];
        }
    }

    /// Encode a flat weight slice, applying `fwht` per 256-group first. A
    /// trailing partial group is zero-padded, matching the uniform encoders.
    pub fn encode(&self, f32_data: &[f32], fwht: impl Fn(&mut [f32]) + Sync) -> Vec<u8> {
        use rayon::prelude::*;
        let n = f32_data.len();
        let n_blocks = n.div_ceil(GROUP);
        let mut out = vec![0u8; n_blocks * self.group_bytes];
        out.par_chunks_mut(self.group_bytes)
            .enumerate()
            .for_each(|(b, chunk)| {
                let start = b * GROUP;
                let end = (start + GROUP).min(n);
                let mut g = [0.0f32; GROUP];
                g[..end - start].copy_from_slice(&f32_data[start..end]);
                fwht(&mut g);
                self.encode_group(&g, chunk);
            });
        out
    }

    /// Decode a whole buffer back to the rotated domain.
    pub fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let n_blocks = bytes.len() / self.group_bytes;
        let mut out = vec![0.0f32; n_blocks * GROUP];
        for b in 0..n_blocks {
            let blk = &bytes[b * self.group_bytes..(b + 1) * self.group_bytes];
            let dst: &mut [f32; GROUP] =
                (&mut out[b * GROUP..(b + 1) * GROUP]).try_into().unwrap();
            self.decode_group(blk, dst);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Spec; 3] = [MQ4N, MQ3N, MQ2N];

    fn gaussian(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16777216.0).clamp(1e-7, 1.0 - 1e-7)
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (next(), next());
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect()
    }

    fn block(seed: u32) -> [f32; GROUP] {
        let mut g = [0.0f32; GROUP];
        for (i, v) in gaussian(GROUP, seed).into_iter().enumerate() {
            g[i] = v;
        }
        g
    }

    /// The uniform min/max fit each MQ*N replaces.
    fn legacy_err(g: &[f32; GROUP], levels: usize) -> f64 {
        let lo = g.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = g.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let scale = if range > 0.0 { range / (levels - 1) as f32 } else { 1.0 };
        let inv = if range > 0.0 { 1.0 / scale } else { 0.0 };
        g.iter()
            .map(|&v| {
                let q = ((v - lo) * inv + 0.5).clamp(0.0, (levels - 1) as f32);
                let d = (v - (q * scale + lo)) as f64;
                d * d
            })
            .sum()
    }

    fn spec_err(spec: &Spec, g: &[f32; GROUP]) -> f64 {
        let mut enc = vec![0u8; spec.group_bytes];
        spec.encode_group(g, &mut enc);
        let mut dec = [0.0f32; GROUP];
        spec.decode_group(&enc, &mut dec);
        g.iter()
            .zip(dec.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum()
    }

    #[test]
    fn byte_budgets_match_the_uniform_formats_they_replace() {
        assert_eq!((MQ4N.group_bytes, MQ4N.payload_bytes()), (136, 128));
        assert_eq!((MQ3N.group_bytes, MQ3N.payload_bytes()), (104, 96));
        assert_eq!((MQ2N.group_bytes, MQ2N.payload_bytes()), (72, 64));
        for s in ALL {
            assert_eq!(s.group_bytes, HEADER + s.payload_bytes());
            assert_eq!(s.codebook.len(), 1usize << s.bits);
        }
    }

    #[test]
    fn codebooks_are_sorted_symmetric_and_unit_peak() {
        for s in ALL {
            let cb = s.codebook;
            for i in 1..cb.len() {
                assert!(cb[i] > cb[i - 1], "{}-bit not ascending at {i}", s.bits);
            }
            for i in 0..cb.len() / 2 {
                assert!(
                    (cb[i] + cb[cb.len() - 1 - i]).abs() < 1e-7,
                    "{}-bit asymmetric at {i}",
                    s.bits
                );
            }
            assert!((cb[cb.len() - 1] - 1.0).abs() < 1e-7);
        }
    }

    /// Packing must be exactly invertible at every width — a bit-order slip in
    /// the cross-byte 3-bit layout is otherwise silent.
    #[test]
    fn pack_unpack_round_trips_every_code() {
        for s in ALL {
            let mut codes = [0u8; GROUP];
            let mask = (1u16 << s.bits) as u8 - 1;
            for i in 0..GROUP {
                codes[i] = ((i as u8).wrapping_mul(37).wrapping_add(11)) & mask;
            }
            let mut payload = vec![0u8; s.payload_bytes()];
            s.pack(&codes, &mut payload);
            let mut back = [0u8; GROUP];
            s.unpack(&payload, &mut back);
            assert_eq!(codes, back, "{}-bit pack/unpack mismatch", s.bits);
        }
    }

    #[test]
    fn nearest_code_is_optimal_at_every_width() {
        for s in ALL {
            for step in 0..2001 {
                let v = -1.6 + 3.2 * (step as f32) / 2000.0;
                let got = s.nearest_code(v) as usize;
                let best = s
                    .codebook
                    .iter()
                    .map(|&c| (c - v).abs())
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    (s.codebook[got] - v).abs() <= best + 1e-9,
                    "{}-bit v={v} picked {got}",
                    s.bits
                );
            }
        }
    }

    #[test]
    fn encode_decode_round_trips_through_the_reference() {
        for s in ALL {
            for seed in 0..8u32 {
                let g = block(seed * 131 + 7);
                let mut enc = vec![0u8; s.group_bytes];
                s.encode_group(&g, &mut enc);
                let mut dec = [0.0f32; GROUP];
                s.decode_group(&enc, &mut dec);
                let master = f32::from_le_bytes([enc[0], enc[1], enc[2], enc[3]]);
                let mut codes = [0u8; GROUP];
                s.unpack(&enc[HEADER..], &mut codes);
                for i in 0..GROUP {
                    let eff = master * e4m3_decode(enc[4 + i / SUB]);
                    assert_eq!(
                        dec[i].to_bits(),
                        (s.codebook[codes[i] as usize] * eff).to_bits(),
                        "{}-bit i={i}",
                        s.bits
                    );
                }
            }
        }
    }

    #[test]
    fn degenerate_groups_are_safe() {
        for s in ALL {
            for probe in [0.0f32, 0.75, -2.5] {
                let g = [probe; GROUP];
                let mut enc = vec![0u8; s.group_bytes];
                s.encode_group(&g, &mut enc);
                let mut dec = [0.0f32; GROUP];
                s.decode_group(&enc, &mut dec);
                for &v in dec.iter() {
                    assert!(
                        (v - probe).abs() <= 1e-6 * probe.abs().max(1.0),
                        "{}-bit constant {probe} -> {v}",
                        s.bits
                    );
                }
            }
        }
    }

    /// The min/max penalty grows as bits shrink, so each width must clear a
    /// progressively higher bar. Floors are set below measured values
    /// (4-bit +2.0, 3-bit +2.5, 2-bit +4.1 dB) to leave headroom.
    #[test]
    fn each_width_beats_its_uniform_min_max_fit() {
        for (s, floor_db) in [(MQ4N, 1.5f64), (MQ3N, 2.0), (MQ2N, 3.5)] {
            let (mut new, mut old) = (0.0f64, 0.0f64);
            for seed in 0..48u32 {
                let g = block(seed * 7919 + 3);
                new += spec_err(&s, &g);
                old += legacy_err(&g, s.codebook.len());
            }
            let gain = 10.0 * (old / new).log10();
            assert!(
                gain > floor_db,
                "{}-bit gained only {gain:.2} dB, expected > {floor_db} dB",
                s.bits
            );
        }
    }

    #[test]
    fn heterogeneous_sub_blocks_pick_distinct_scales() {
        for s in ALL {
            let mut g = [0.0f32; GROUP];
            for (i, v) in gaussian(GROUP, 99).into_iter().enumerate() {
                g[i] = v * [1.0f32, 0.05, 8.0, 0.4][i / SUB];
            }
            let mut enc = vec![0u8; s.group_bytes];
            s.encode_group(&g, &mut enc);
            let distinct = enc[4..8].iter().collect::<std::collections::HashSet<_>>().len();
            assert!(distinct >= 3, "{}-bit sub-scales collapsed: {:?}", s.bits, &enc[4..8]);
        }
    }

    #[test]
    fn encode_produces_whole_groups() {
        for s in ALL {
            let data = gaussian(GROUP * 5, 3);
            assert_eq!(s.encode(&data, |_| {}).len(), 5 * s.group_bytes);
        }
    }
}
