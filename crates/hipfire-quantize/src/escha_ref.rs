//! Portable CPU reference for the escha codec — the numerical oracle for
//! every GPU kernel in this port.
//!
//! Ported from `escha_mlx/ref.py` (EschaLabs/escha-mlx, Apache-2.0), which
//! declares itself "the semantic contract for every Metal kernel in this
//! package". Rounding points are deliberate; do not "simplify" them.

use crate::float16::f16_to_f32;

/// 1/sqrt(128) — the exact f32 constant the format pins.
pub const RS: f32 = 0.088388347648;

/// Round `v` to fp16 bits using round-to-nearest-even.
///
/// This is the one RNE encode site in the module. `crate::float16::f32_to_f16`
/// is documented to truncate ("Hipfire's historical truncating conversion")
/// to keep existing HFQ encoder output byte-stable, whereas every `f16(...)`
/// in the escha contract is RNE — truncating flips the low bit on roughly
/// half of all 65536 codebook states (state 3 alone: truncation gives
/// 0x3ab7, the published/measured value is 0x3ab8). `half` is already a
/// workspace dependency used inside `float16.rs` itself, so this is not a
/// new dependency, just reaching past the crate's non-RNE convenience
/// wrapper for the one place where RNE is the spec.
#[inline]
pub fn f16_rne(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Decode one 16-bit trellis state to fp16 **bits** via the cbA codebook.
///
/// `decode(x) = f16_lo(r) + f16_hi(r)` with fp16 RNE addition, where
/// `r = ((x * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60` in 32-bit arithmetic.
///
/// Adding in f32 and rounding once is exactly an fp16 RNE add: the exact sum
/// of two fp16 values is always representable in f32, so the single rounding
/// here is the correctly-rounded fp16 result.
///
/// There are 65536 reachable values, so a lookup table would be 128 KB and
/// will not fit gfx1151's 64 KB LDS. This is five integer/FP ops and no
/// memory traffic — keep it that way in the kernels.
#[inline]
pub fn cba_decode(state: u16) -> u16 {
    let r = ((state as u32).wrapping_mul(0xCBAC_1FED) & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    let lo = f16_to_f32((r & 0xFFFF) as u16);
    let hi = f16_to_f32((r >> 16) as u16);
    f16_rne(lo + hi)
}

/// The 8 states lane `lane` owns, K=2. `words` is the tile's 16 u32.
///
/// DELIBERATE DUPLICATION: `kernels/src/escha_decode_tiles.hip` implements
/// this same lane maths independently. That is the G2 gate — the GPU decode
/// is asserted bit-exact against this one. Generating either from the other,
/// or sharing a source, would make G2 circular: both paths could be wrong in
/// exactly the same way and still agree. Two independent implementations of
/// a published spec is the point. Do not deduplicate.
pub fn decode8_k2(words: &[u32; 16], lane: usize) -> [u16; 8] {
    let t_off = lane * 8;
    let i1 = t_off >> 4;
    let i0 = (i1 + 15) & 15;
    let merged = ((words[i0] as u64) << 32) | words[i1] as u64;
    let shift = ((!t_off) & 8) << 1; // 16 for even lanes, 0 for odd
    let w = ((merged >> shift) & 0xFFFF_FFFF) as u32;
    let mut out = [0u16; 8];
    for (j, o) in out.iter_mut().enumerate() {
        *o = (w >> (2 * (7 - j))) as u16;
    }
    out
}

/// The 8 states lane `lane` owns, K=3. `words` is the tile's 24 u32.
///
/// Structurally different from K=2 — 24 words, a computed bit offset, and a
/// modular wrap. Do not attempt to unify the two.
pub fn decode8_k3(words: &[u32; 24], lane: usize) -> [u16; 8] {
    const BITS: usize = 3;
    let t_off = lane * 8;
    let b1 = (t_off + 257) * BITS;
    let b0 = b1 - 16;
    let b2 = b1 + BITS * 7;
    let i0 = b0 >> 5;
    let i2 = (b2 - 1) >> 5;
    let s2 = ((i2 + 1) << 5) - b2;
    let merged = ((words[i0 % 24] as u64) << 32) | words[i2 % 24] as u64;
    let w7 = (merged >> s2) & 0xFFFF_FFFF;
    let w3 = (merged >> (s2 + BITS * 4)) & 0xFFFF_FFFF;
    [
        (w3 >> 9) as u16,
        (w3 >> 6) as u16,
        (w3 >> 3) as u16,
        w3 as u16,
        (w7 >> 9) as u16,
        (w7 >> 6) as u16,
        (w7 >> 3) as u16,
        w7 as u16,
    ]
}

/// `(row, col)` inside the 16x16 tile for each of the lane's 8 values.
///
/// This permutation is the single easiest thing to get subtly wrong: a wrong
/// shuffle still yields a full-rank, plausible weight matrix. It is gated
/// directly on golden vectors, never on end-to-end coherence.
pub fn lane_positions(lane: usize) -> [(usize, usize); 8] {
    let l0 = lane & !4;
    let c_off = (lane >> 2) & 1;
    let mut out = [(0usize, 0usize); 8];
    for (j, o) in out.iter_mut().enumerate() {
        let fi = j >> 1;
        let row = (lane & 3) * 2 + (j & 1) + (fi & 1) * 8;
        let col = 2 * ((l0 >> 3) + if j >= 4 { 4 } else { 0 }) + c_off;
        *o = (row, col);
    }
    out
}

/// Decode one packed tile (`16*K` i16) to a 16x16 fp16-bit tile, row-major.
pub fn decode_tile(tile: &[i16], k: usize) -> [u16; 256] {
    debug_assert_eq!(tile.len(), 16 * k);
    let mut words = [0u32; 24];
    for (i, w) in words.iter_mut().enumerate().take(8 * k) {
        *w = (tile[2 * i] as u16 as u32) | ((tile[2 * i + 1] as u16 as u32) << 16);
    }
    let mut out = [0u16; 256];
    for lane in 0..32 {
        let states = match k {
            2 => {
                let mut w16 = [0u32; 16];
                w16.copy_from_slice(&words[..16]);
                decode8_k2(&w16, lane)
            }
            3 => decode8_k3(&words, lane),
            _ => panic!("unsupported escha K={k}"),
        };
        for (j, (r, c)) in lane_positions(lane).into_iter().enumerate() {
            out[r * 16 + c] = cba_decode(states[j]);
        }
    }
    out
}

/// Packed `(in/16, out/16, 16K)` i16 -> `(in, out)` fp16 bits, row-major.
pub fn reconstruct(code: &[i16], in_features: usize, out_features: usize, k: usize) -> Vec<u16> {
    let (tk, tn) = (in_features / 16, out_features / 16);
    assert_eq!(code.len(), tk * tn * 16 * k, "escha code length mismatch");
    let mut out = vec![0u16; in_features * out_features];
    for kt in 0..tk {
        for nt in 0..tn {
            let base = (kt * tn + nt) * 16 * k;
            let tile = decode_tile(&code[base..base + 16 * k], k);
            for r in 0..16 {
                let dst = (kt * 16 + r) * out_features + nt * 16;
                out[dst..dst + 16].copy_from_slice(&tile[r * 16..r * 16 + 16]);
            }
        }
    }
    out
}

/// Unnormalised 128-point Walsh-Hadamard (Sylvester / natural order), applied
/// independently to each contiguous 128-element block of `x`.
///
/// `x.len()` must be a multiple of 128. Every dimension in both checkpoints
/// satisfies this (512, 1024, 2048, 5120, 6144, 10240, 17408 are all
/// multiples of 128), including the gate|up split point at 512 — so a block
/// never straddles the gate/up boundary.
pub fn h128_inplace(x: &mut [f32]) {
    assert_eq!(
        x.len() % 128,
        0,
        "H128 needs a multiple of 128, got {}",
        x.len()
    );
    for block in x.chunks_exact_mut(128) {
        let mut h = 1;
        while h < 128 {
            let mut i = 0;
            while i < 128 {
                for j in i..i + h {
                    let (a, b) = (block[j], block[j + h]);
                    block[j] = a + b;
                    block[j + h] = a - b;
                }
                i += 2 * h;
            }
            h *= 2;
        }
    }
}

/// `xh = f16( H128(x * rin) * RS )`. Returns fp16 bits.
pub fn input_transform(x: &[f32], rin: &[f32]) -> Vec<u16> {
    let ic = rin.len();
    assert_eq!(x.len() % ic, 0);
    let mut buf: Vec<f32> = x
        .iter()
        .zip(rin.iter().cycle())
        .map(|(a, b)| a * b)
        .collect();
    for row in buf.chunks_exact_mut(ic) {
        h128_inplace(row);
    }
    buf.iter().map(|v| f16_rne(v * RS)).collect()
}

/// `y = f16( H128(mid) * RS * rout )`. Returns fp16 bits.
pub fn output_transform(mid: &[f32], rout: &[f32]) -> Vec<u16> {
    let oc = rout.len();
    assert_eq!(mid.len() % oc, 0);
    let mut buf = mid.to_vec();
    for row in buf.chunks_exact_mut(oc) {
        h128_inplace(row);
    }
    buf.iter()
        .zip(rout.iter().cycle())
        .map(|(v, s)| f16_rne(v * RS * s))
        .collect()
}

/// Fold the optional end-to-end scales into the transform vectors.
///
/// `s_in` multiplies the activation at exactly the point `rin` does, and
/// `s_out` at exactly the point `rout` does, so the pair collapses with no new
/// kernel and no new tensor. Folding keeps both products in f32 and rounds
/// once — one rounding point FEWER than applying the scales separately.
/// `None` returns that vector unchanged (as f32), which is the path MoE
/// exports and end-to-end-free exports both take.
pub fn fold_scales(
    rin: &[u16],
    rout: &[u16],
    s_in: Option<&[f32]>,
    s_out: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let mut ri: Vec<f32> = rin.iter().map(|&b| f16_to_f32(b)).collect();
    let mut ro: Vec<f32> = rout.iter().map(|&b| f16_to_f32(b)).collect();
    if let Some(s) = s_in {
        assert_eq!(s.len(), ri.len());
        for (a, b) in ri.iter_mut().zip(s) {
            *a *= b;
        }
    }
    if let Some(s) = s_out {
        assert_eq!(s.len(), ro.len());
        for (a, b) in ro.iter_mut().zip(s) {
            *a *= b;
        }
    }
    (ri, ro)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/escha")
            .join(name)
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The codebook is a pure function of the 16-bit state. These eight values
    /// were computed from the published constants and pin the hash, the
    /// masking, and the fp16 RNE add all at once.
    #[test]
    fn cba_decode_matches_published_constants() {
        let want: [u16; 8] = [
            0x3f60, 0x304e, 0xba13, 0x3ab8, 0x3952, 0xb75f, 0xbea4, 0xbc71,
        ];
        for (state, &bits) in want.iter().enumerate() {
            assert_eq!(cba_decode(state as u16), bits, "state {state}");
        }
    }

    #[test]
    fn reconstruct_k2_matches_golden() {
        let raw = std::fs::read(data("packed_gu_e0_k2.i16")).unwrap();
        let code: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(code.len(), 262144);
        let out = reconstruct(&code, 2048, 1024, 2);
        assert_eq!(out.len(), 2048 * 1024);
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            sha256_hex(&bytes),
            "51ddde9a07613aafcc9f5db79702349d19e18357d9fb910f48b38eeea028dcab",
            "decoded K=2 tensor does not match expected_gu_e0_k2.f16"
        );
    }

    #[test]
    fn reconstruct_k3_matches_golden() {
        let raw = std::fs::read(data("packed_down_e0_k3.i16")).unwrap();
        let code: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(code.len(), 196608);
        let out = reconstruct(&code, 512, 2048, 3);
        assert_eq!(out.len(), 512 * 2048);
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            sha256_hex(&bytes),
            "51c99817d00282f4aa9d618140eb4503083e1238cb59dd71a670aa4c320f7438",
            "decoded K=3 tensor does not match expected_down_e0_k3.f16"
        );
    }

    /// Every one of the 256 tile slots must be written exactly once. A
    /// permutation bug that drops and duplicates slots still produces a
    /// full-rank, plausible-looking matrix, so check the permutation directly.
    #[test]
    fn lane_positions_is_a_permutation_of_the_tile() {
        let mut seen = [0u8; 256];
        for lane in 0..32 {
            for (r, c) in lane_positions(lane) {
                assert!(r < 16 && c < 16, "lane {lane} -> ({r},{c})");
                seen[r * 16 + c] += 1;
            }
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "lane_positions is not a bijection"
        );
    }

    /// H128 is its own inverse up to a factor of 128. That is necessary but
    /// NOT sufficient: a wrong butterfly order is also self-inverse and would
    /// pass this alone. The Hadamard-of-a-basis-vector check below pins the
    /// actual transform.
    #[test]
    fn h128_roundtrip_scales_by_128() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
        let orig = x.clone();
        h128_inplace(&mut x);
        h128_inplace(&mut x);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b * 128.0).abs() < 1e-2, "{a} vs {}", b * 128.0);
        }
    }

    /// H128 applied to e_0 must give all ones (Sylvester, unnormalised).
    /// Applied to e_1 it must give the alternating +1/-1 pattern of row 1.
    #[test]
    fn h128_matches_sylvester_order() {
        let mut e0 = vec![0.0f32; 128];
        e0[0] = 1.0;
        h128_inplace(&mut e0);
        assert!(e0.iter().all(|&v| v == 1.0), "row 0 must be all ones");

        let mut e1 = vec![0.0f32; 128];
        e1[1] = 1.0;
        h128_inplace(&mut e1);
        for (i, &v) in e1.iter().enumerate() {
            let want = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(v, want, "index {i}");
        }
    }

    /// Blocks are independent: H128 must never mix across a 128 boundary.
    #[test]
    fn h128_does_not_mix_across_blocks() {
        let mut x = vec![0.0f32; 256];
        x[0] = 1.0;
        h128_inplace(&mut x);
        assert!(x[..128].iter().all(|&v| v == 1.0));
        assert!(
            x[128..].iter().all(|&v| v == 0.0),
            "second block was contaminated"
        );
    }

    /// MoE exports ship all-ones s_in/s_out; dense exports ship real values;
    /// and an export without the end-to-end stage ships neither. All three
    /// must go through one code path.
    #[test]
    fn fold_scales_handles_absent_scales() {
        let rin = [f16_rne(2.0), f16_rne(-3.0)];
        let rout = [f16_rne(0.5)];
        let (a, b) = fold_scales(&rin, &rout, None, None);
        assert_eq!(a, vec![2.0, -3.0]);
        assert_eq!(b, vec![0.5]);
        let (c, d) = fold_scales(&rin, &rout, Some(&[3.0, 2.0]), Some(&[4.0]));
        assert_eq!(c, vec![6.0, -6.0]);
        assert_eq!(d, vec![2.0]);
    }
}
