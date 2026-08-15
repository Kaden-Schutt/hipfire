// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — E8 lattice codec for mfp4-E8 (DType MFP4G32E8).
//
// E8 = D8 ∪ (D8 + 1/2·1),  D8 = { x ∈ Z^8 : sum(x) even }.
// Operates on block-NORMALIZED weight vectors v[0..8] in ≈[-6,6].
// QUANT_STEP = 0.88 (constant, tuned to the FWHT block-normalized domain).
//
// 32-bit index bijection (bit-exact CPU↔kernel):
//   bits[4i .. 4i+4) = e[i]  for i=0..6   (28 bits)
//   bits[28..31)     = e[7]>>1 & 0x7       (high 3 bits, LSB dropped)
//   bit[31]          = coset                (0=D8, 1=D8+1/2)
// Total = 32 bits.  Decode recovers e[7] LSB from parity of sum(e[0..7]).

pub const QUANT_STEP: f32 = 0.88;
/// E8_SIGMA: std-dev of the FWHT-normalized weight distribution (used in tests).
pub const E8_SIGMA: f32 = 1.7;

// --- n-bit generalized E8 QUANT_STEP (mfp3=3-bit, mfp2=2-bit).
// FINALIZED via the joint (q, bias) packing-gain sweep (Test E) on the
// FWHT-normalized Gaussian (σ=1.7). The recenter bias was tuned jointly with q
// (see coord_bias_n below: n=3→4, n=2→2) because the coset's +0.5 offset makes
// the box lean the wrong way at the default 2^(n-1)-1 bias — re-biasing cut MSE
// 27% (mfp3) and 50% (mfp2) and roughly halved clamp saturation. The locked q is
// the interior minimum of the bowl at the chosen bias (cross-seed stable ±0.1%).
// These MUST match the kernel-side #define QUANT_STEP_MFP3 / _MFP2 in every .hip
// file AND the MFP3_STEP / MFP2_STEP consts in qwen35.rs CPU dequant.
/// QUANT_STEP for mfp3-E8 (3-bit nibbles, box [-4,3] via bias=4). MSE-optimal q.
pub const QUANT_STEP_MFP3: f32 = 1.8;
/// QUANT_STEP for mfp2-E8 (2-bit nibbles, box [-2,1] via bias=2). MSE-optimal q.
pub const QUANT_STEP_MFP2: f32 = 3.8;

const COORD_BIAS: i32 = 7; // biased range [0,15] → integer range [-7,8]
const COORD_BITS: u32 = 4;

// --------------------------------------------------------------------------
// Round-to-nearest, ties away from zero (integer-deterministic, matches kernel).
// --------------------------------------------------------------------------
#[inline(always)]
fn round_tie_away(x: f32) -> f32 {
    if x >= 0.0 {
        (x + 0.5).floor()
    } else {
        (x - 0.5).ceil()
    }
}

// --------------------------------------------------------------------------
// Nearest point in D8 (Conway-Sloane).
// --------------------------------------------------------------------------
fn closest_d8(u: &[f32; 8]) -> [f32; 8] {
    let mut r = [0.0f32; 8];
    let mut s: i64 = 0;
    let mut worst_idx = 0usize;
    let mut worst_abs = -1.0f32;
    let mut worst_dir = 0.0f32;
    for i in 0..8 {
        let ri = round_tie_away(u[i]);
        r[i] = ri;
        s += ri as i64;
        let e = u[i] - ri;
        let a = e.abs();
        if a > worst_abs {
            worst_abs = a;
            worst_idx = i;
            // e >= 0 means ri was rounded DOWN; flip UP (+1).
            worst_dir = if e >= 0.0 { 1.0 } else { -1.0 };
        }
    }
    // Fix parity: if sum is odd, flip the coord with largest fractional distance.
    if (s & 1) != 0 {
        r[worst_idx] += worst_dir;
    }
    r
}

// --------------------------------------------------------------------------
// Nearest point in E8 = min(closest in D8, closest in D8+1/2).
// --------------------------------------------------------------------------
fn closest_e8(u: &[f32; 8]) -> [f32; 8] {
    let a = closest_d8(u);
    // Shift for the D8+1/2 coset.
    let mut ush = [0.0f32; 8];
    for i in 0..8 {
        ush[i] = u[i] - 0.5;
    }
    let bsh = closest_d8(&ush);
    let mut b = [0.0f32; 8];
    for i in 0..8 {
        b[i] = bsh[i] + 0.5;
    }

    let da: f32 = (0..8)
        .map(|i| {
            let e = u[i] - a[i];
            e * e
        })
        .sum();
    let db: f32 = (0..8)
        .map(|i| {
            let e = u[i] - b[i];
            e * e
        })
        .sum();
    if da <= db {
        a
    } else {
        b
    }
}

// --------------------------------------------------------------------------
// Encode: E8 point → u32 index.
// --------------------------------------------------------------------------
pub fn encode_index(p: &[f32; 8]) -> u32 {
    // Determine coset: half-integer coords → coset=1.
    let coset = if (p[0].fract().abs() - 0.5).abs() < 0.1 {
        1u32
    } else {
        0u32
    };

    // Integer coords of the underlying D8 point.
    let mut w = [0i32; 8];
    for i in 0..8 {
        w[i] = if coset == 1 {
            (p[i] - 0.5).round() as i32
        } else {
            p[i].round() as i32
        };
    }

    // Bias and clamp to [0,15].
    let mut e = [0u32; 8];
    for i in 0..8 {
        e[i] = (w[i] + COORD_BIAS).clamp(0, 15) as u32;
    }

    // Re-establish even sum (may have been broken by clamp).
    let sl: u32 = e.iter().sum();
    if (sl & 1) != 0 {
        // Nudge e[7] by ±1 within [0,15].
        if e[7] < 15 {
            e[7] += 1;
        } else {
            e[7] -= 1;
        }
    }

    // Pack: bits[4i..4i+4) = e[i] for i=0..6, bits[28..31) = e[7]>>1, bit[31]=coset.
    let mut idx: u32 = 0;
    for i in 0..7 {
        idx |= (e[i] & 0xF) << (i as u32 * COORD_BITS);
    }
    idx |= ((e[7] >> 1) & 0x7) << 28;
    idx |= coset << 31;
    idx
}

// --------------------------------------------------------------------------
// Decode: u32 index → E8 point (f32[8]).
// --------------------------------------------------------------------------
pub fn decode_index(idx: u32) -> [f32; 8] {
    let coset = (idx >> 31) & 1;
    let mut e = [0u32; 8];
    let mut sl: u32 = 0;
    for i in 0..7 {
        e[i] = (idx >> (i as u32 * COORD_BITS)) & 0xF;
        sl += e[i];
    }
    let e7_high = (idx >> 28) & 0x7;
    let p7 = e7_high << 1;
    // Recover LSB: parity of (sl + p7) must be even (sum of all 8 biased coords even).
    let lsb = (sl + p7) & 1;
    e[7] = p7 | lsb;

    let mut p = [0.0f32; 8];
    for i in 0..8 {
        let c = (e[i] as i32 - COORD_BIAS) as f32;
        p[i] = if coset == 1 { c + 0.5 } else { c };
    }
    p
}

// --------------------------------------------------------------------------
// Public API: quantize 8 weights → u32, dequantize u32 → 8 f32.
// --------------------------------------------------------------------------

/// Quantize 8 block-normalized weights (range ≈[-6,6]) to a 32-bit E8 codeword.
/// `q` = QUANT_STEP (0.88); pass `QUANT_STEP` constant.
pub fn quantize8(v: &[f32; 8], q: f32) -> u32 {
    let mut u = [0.0f32; 8];
    for i in 0..8 {
        u[i] = v[i] / q;
    }
    let p = closest_e8(&u);
    encode_index(&p)
}

/// Dequantize a 32-bit E8 codeword back to 8 f32 values (block-normalized domain).
/// `q` = QUANT_STEP (0.88).
pub fn dequantize8(idx: u32, q: f32) -> [f32; 8] {
    let p = decode_index(idx);
    let mut v = [0.0f32; 8];
    for i in 0..8 {
        v[i] = p[i] * q;
    }
    v
}

// --------------------------------------------------------------------------
// n-bit generalized E8 codec  (n=2, 3, 4; n=4 is byte-identical to above).
//
// Layout per n-bit codeword (`8n` bits packed LE):
//   bits[n·i .. n·i+n) = e[i]        for i = 0..6   (7·n bits)
//   bits[7·n .. 8·n-1) = e[7]>>1     high (n-1) bits of biased e[7]
//   bit [8·n-1]        = coset        (0 = D8, 1 = D8+1/2)
// e[7] LSB is reconstructed from the even-sum (D8) parity constraint.
//
// Constant summary by n:
//   COORD_BIAS_n (recenter)     → 4: 7   3: 4   2: 2   (TUNED, see below)
//   clamp max    = 2^n - 1      → 4: 15  3: 7   2: 3
//   coset bit    = 8n-1         → 4: 31  3: 23  2: 15
//   codeword bytes = n          → 4: 4   3: 3   2: 2
//
// The lattice search (closest_d8 / closest_e8) runs on continuous floats and
// is fully n-independent — the only n-dependent part is the clamped coordinate box.
//
// RECENTER BIAS is NOT the textbook 2^(n-1)-1 (which gives 7/3/1). At low n the
// E8+½ coset's +0.5 shift makes a symmetric Gaussian lean past the box, piling the
// negative tail on the floor. A joint (q,bias) MSE sweep on the FWHT-Gaussian picks
// bias = {n=3: 4 (box [-4,3]), n=2: 2 (box [-2,1])} — cutting MSE 27%/50% and
// roughly halving clamp saturation vs 3/1. n=4 stays 7 (mfp4-E8 unchanged).
// The on-disk format depends on this value: it MUST stay in lockstep with the
// kernel decoders' center literal (-4 / -2) and qwen35.rs CPU dequant.
// --------------------------------------------------------------------------

#[inline(always)]
fn coord_bias_n(n: u32) -> i32 {
    // Tuned recenter (see header). NOT 2^(n-1)-1 for n<4.
    match n {
        4 => 7, // mfp4-E8: unchanged (box [-7,8])
        3 => 4, // mfp3-E8: box [-4,3]  (was 3 → [-3,4]; re-biased, MSE -27%)
        2 => 2, // mfp2-E8: box [-2,1]  (was 1 → [-1,2]; re-biased, MSE -50%)
        _ => (1i32 << (n - 1)) - 1,
    }
}

#[inline(always)]
fn coord_max_n(n: u32) -> i32 {
    (1i32 << n) - 1
} // n=4→15, n=3→7, n=2→3

/// Encode an E8 lattice point to an `8n`-bit codeword (returned in low `8n` bits of u32).
/// `p` must be an actual E8 point (integer or half-integer coords, even D8 sum).
/// The upper `32 - 8n` bits of the returned u32 are ZERO — this is required for
/// the narrow codeword reads in the kernels (byte-assembly over `n` bytes).
pub fn encode_index_n(p: &[f32; 8], n: u32) -> u32 {
    let bias = coord_bias_n(n);
    let cmax = coord_max_n(n);
    let mask: u32 = (1u32 << n) - 1;

    // Detect coset from fractional part of the first coordinate.
    let coset = if (p[0].fract().abs() - 0.5).abs() < 0.1 {
        1u32
    } else {
        0u32
    };

    // Integer coords of underlying D8 point.
    let mut w = [0i32; 8];
    for i in 0..8 {
        w[i] = if coset == 1 {
            (p[i] - 0.5).round() as i32
        } else {
            p[i].round() as i32
        };
    }

    // Bias, clamp to [0, 2^n - 1].
    let mut e = [0u32; 8];
    for i in 0..8 {
        e[i] = (w[i] + bias).clamp(0, cmax) as u32;
    }

    // Re-establish even sum (clamping may have broken parity).
    let sl: u32 = e.iter().sum();
    if (sl & 1) != 0 {
        // Nudge e[7] by ±1 within [0, cmax]; same semantics as the n=4 path.
        if (e[7] as i32) < cmax {
            e[7] += 1;
        } else {
            e[7] -= 1;
        }
    }

    // Pack: bits[n·i .. n·i+n) = e[i] for i=0..6;
    //       bits[7n .. 8n-1)   = e[7]>>1 (high n-1 bits);
    //       bit[8n-1]          = coset.
    let mut idx: u32 = 0;
    for i in 0..7_u32 {
        idx |= (e[i as usize] & mask) << (i * n);
    }
    let e7_high_bits = n - 1;
    let e7_high_mask: u32 = (1u32 << e7_high_bits) - 1;
    idx |= ((e[7] >> 1) & e7_high_mask) << (7 * n);
    idx |= coset << (8 * n - 1);
    idx
}

/// Decode an `8n`-bit codeword (low `8n` bits of `idx`, upper bits must be zero)
/// to an E8 lattice point.
pub fn decode_index_n(idx: u32, n: u32) -> [f32; 8] {
    let bias = coord_bias_n(n);
    let mask: u32 = (1u32 << n) - 1;

    let coset = (idx >> (8 * n - 1)) & 1;
    let mut e = [0u32; 8];
    let mut sl: u32 = 0;
    for i in 0..7_u32 {
        e[i as usize] = (idx >> (i * n)) & mask;
        sl += e[i as usize];
    }
    // Recover e[7]: high (n-1) bits come from the packed field; LSB from parity.
    let e7_high_bits = n - 1;
    let e7_high_mask: u32 = (1u32 << e7_high_bits) - 1;
    let e7_high = (idx >> (7 * n)) & e7_high_mask;
    let p7 = e7_high << 1;
    // e[7] LSB: the sum of all 8 biased coords must be even.
    e[7] = p7 | ((sl + p7) & 1);

    let mut p = [0.0f32; 8];
    for i in 0..8 {
        let c = (e[i] as i32 - bias) as f32;
        p[i] = if coset == 1 { c + 0.5 } else { c };
    }
    p
}

/// Quantize 8 block-normalized weights using the n-bit E8 codec.
/// `q` = QUANT_STEP_MFP3 (n=3) or QUANT_STEP_MFP2 (n=2).
pub fn quantize8_n(v: &[f32; 8], q: f32, n: u32) -> u32 {
    let mut u = [0.0f32; 8];
    for i in 0..8 {
        u[i] = v[i] / q;
    }
    let p = closest_e8(&u);
    encode_index_n(&p, n)
}

/// Dequantize an n-bit E8 codeword to 8 f32 values.
pub fn dequantize8_n(idx: u32, q: f32, n: u32) -> [f32; 8] {
    let p = decode_index_n(idx, n);
    let mut v = [0.0f32; 8];
    for i in 0..8 {
        v[i] = p[i] * q;
    }
    v
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Simple deterministic LCG for test randomness (no external dep).
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let hi = (*state >> 32) as u32;
        (hi as f32 / (u32::MAX as f32)) * 2.0 - 1.0
    }

    fn box_muller(state: &mut u64, sigma: f32) -> (f32, f32) {
        loop {
            let u1 = (lcg_next(state) + 1.0) * 0.5; // uniform (0,1]
            let u2 = (lcg_next(state) + 1.0) * 0.5;
            if u1 <= 0.0 {
                continue;
            }
            let r = (-2.0 * u1.ln()).sqrt() * sigma;
            let theta = 2.0 * std::f32::consts::PI * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }

    #[test]
    fn round_tie_basic() {
        assert_eq!(round_tie_away(0.5), 1.0);
        assert_eq!(round_tie_away(-0.5), -1.0);
        assert_eq!(round_tie_away(1.4), 1.0);
        assert_eq!(round_tie_away(1.6), 2.0);
        assert_eq!(round_tie_away(-1.4), -1.0);
        assert_eq!(round_tie_away(-1.6), -2.0);
    }

    #[test]
    fn d8_even_sum() {
        let mut state = 0x12345678u64;
        for _ in 0..2000 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = lcg_next(&mut state) * 6.0;
            }
            let p = closest_d8(&v);
            let s: i32 = p.iter().map(|&x| x as i32).sum();
            assert_eq!(s & 1, 0, "D8 sum not even: {:?}", p);
        }
    }

    #[test]
    fn index_roundtrip() {
        let mut state = 0xdeadbeef_cafebabeu64;
        let mut failures = 0usize;
        for _ in 0..500_000 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = lcg_next(&mut state) * 6.0;
            }
            let idx = quantize8(&v, QUANT_STEP);
            // Re-encode the decoded point — must give the same index.
            let p = decode_index(idx);
            let idx2 = encode_index(&p);
            if idx != idx2 {
                failures += 1;
            }
            // dequantize8 must equal decode_index * QUANT_STEP
            let dq = dequantize8(idx, QUANT_STEP);
            for i in 0..8 {
                let expected = p[i] * QUANT_STEP;
                assert!(
                    (dq[i] - expected).abs() < 1e-6,
                    "dequantize8 mismatch at i={i}: {:.6} vs {:.6}",
                    dq[i],
                    expected
                );
            }
        }
        assert_eq!(failures, 0, "bijection failures: {failures}");
    }

    /// Packing-gain gate: E8 MSE must beat E2M1 scalar MSE at the same block normalization.
    /// At q=0.88 on FWHT-normalized Gaussian, ratio should be ~1.298.
    #[test]
    fn packing_gain_beats_e2m1() {
        const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        fn e2m1_nearest(x: f32) -> f32 {
            let mut best = 0.0f32;
            let mut best_d = f32::MAX;
            for &m in &E2M1_MAG {
                for &s in &[1.0f32, -1.0f32] {
                    let v = m * s;
                    let d = (x - v).abs();
                    if d < best_d {
                        best_d = d;
                        best = v;
                    }
                }
            }
            best
        }

        let mut state = 0xabcdef0123456789u64;
        let sigmas = [1.0f32, 1.5, E8_SIGMA, 2.0, 2.5];

        for &sigma in &sigmas {
            let n_blocks = 62500usize; // 500k weights / 8
            let mut e8_mse = 0.0f64;
            let mut e2_mse = 0.0f64;
            let mut count = 0usize;

            for _ in 0..n_blocks {
                let mut block = [0.0f32; 8];
                for i in 0..4 {
                    let (a, b) = box_muller(&mut state, sigma);
                    block[2 * i] = a;
                    block[2 * i + 1] = b;
                }
                // Per-block normalization: max/6 (same as mfp4+P).
                let bmax = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
                if bmax == 0.0 {
                    continue;
                }
                let sc = bmax / 6.0;
                let mut vn = [0.0f32; 8];
                for i in 0..8 {
                    vn[i] = block[i] / sc;
                }

                // E8 MSE in original domain.
                let dq = dequantize8(quantize8(&vn, QUANT_STEP), QUANT_STEP);
                for i in 0..8 {
                    let err = (vn[i] - dq[i]) * sc;
                    e8_mse += (err * err) as f64;
                }

                // E2M1 MSE in original domain.
                for i in 0..8 {
                    let nearest = e2m1_nearest(vn[i]);
                    let err = (vn[i] - nearest) * sc;
                    e2_mse += (err * err) as f64;
                }
                count += 8;
            }

            let e8_avg = e8_mse / count as f64;
            let e2_avg = e2_mse / count as f64;
            let ratio = e2_avg / e8_avg;
            eprintln!("sigma={sigma:.1}: E8 {e8_avg:.6} E2M1 {e2_avg:.6} ratio {ratio:.3}");
            assert!(
                e8_avg < e2_avg,
                "E8 packing gain FAILED at sigma={sigma}: e8_mse={e8_avg:.6} >= e2m1_mse={e2_avg:.6}"
            );
            assert!(
                ratio > 1.1 && ratio < 1.6,
                "E8/E2M1 MSE ratio out of expected range at sigma={sigma}: {ratio:.3}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test A — bijection (n=2, n=3): encode_index_n ∘ decode_index_n = id,
    //           dequantize8_n ≡ decode_index_n · q.
    //           Also verifies high-bit invariant: bits above 8n are zero.
    // -----------------------------------------------------------------------
    #[test]
    fn n_bit_bijection_n3() {
        let mut state = 0x1234_5678_abcd_ef01u64;
        let n = 3u32;
        let q = QUANT_STEP_MFP3;
        let mut failures = 0usize;
        for _ in 0..500_000 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = lcg_next(&mut state) * 6.0;
            }
            let idx = quantize8_n(&v, q, n);

            // High bits must be zero (required for narrow kernel reads).
            assert_eq!(
                idx >> (8 * n),
                0,
                "n=3: high bits non-zero: idx=0x{idx:08x}"
            );

            let p = decode_index_n(idx, n);
            let idx2 = encode_index_n(&p, n);
            if idx != idx2 {
                failures += 1;
            }

            // dequantize8_n must equal decode_index_n * q.
            let dq = dequantize8_n(idx, q, n);
            for i in 0..8 {
                let expected = p[i] * q;
                assert!(
                    (dq[i] - expected).abs() < 1e-6,
                    "n=3 dequantize8_n mismatch i={i}: {:.6} vs {:.6}",
                    dq[i],
                    expected
                );
            }
        }
        assert_eq!(failures, 0, "n=3 bijection failures: {failures}");
    }

    #[test]
    fn n_bit_bijection_n2() {
        let mut state = 0xfeed_face_dead_beefu64;
        let n = 2u32;
        let q = QUANT_STEP_MFP2;
        let mut failures = 0usize;
        for _ in 0..500_000 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = lcg_next(&mut state) * 6.0;
            }
            let idx = quantize8_n(&v, q, n);

            // High bits must be zero.
            assert_eq!(
                idx >> (8 * n),
                0,
                "n=2: high bits non-zero: idx=0x{idx:08x}"
            );

            let p = decode_index_n(idx, n);
            let idx2 = encode_index_n(&p, n);
            if idx != idx2 {
                failures += 1;
            }

            let dq = dequantize8_n(idx, q, n);
            for i in 0..8 {
                let expected = p[i] * q;
                assert!(
                    (dq[i] - expected).abs() < 1e-6,
                    "n=2 dequantize8_n mismatch i={i}: {:.6} vs {:.6}",
                    dq[i],
                    expected
                );
            }
        }
        assert_eq!(failures, 0, "n=2 bijection failures: {failures}");
    }

    // Verify n=4 path through encode_index_n / decode_index_n matches n=4 encode/decode.
    #[test]
    fn n4_matches_n_equals_4() {
        let mut state = 0x0102_0304_0506_0708u64;
        let n = 4u32;
        for _ in 0..100_000 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = lcg_next(&mut state) * 6.0;
            }
            let idx_ref = quantize8(&v, QUANT_STEP);
            let idx_n = quantize8_n(&v, QUANT_STEP, n);
            assert_eq!(idx_ref, idx_n, "n=4 mismatch for v={v:?}");
            let p_ref = decode_index(idx_ref);
            let p_n = decode_index_n(idx_ref, n);
            for i in 0..8 {
                assert!(
                    (p_ref[i] - p_n[i]).abs() < 1e-7,
                    "n=4 decode mismatch i={i}: {:.7} vs {:.7}",
                    p_ref[i],
                    p_n[i]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test B — lattice validity after decode.
    //   Coset 0 (D8): all coords integer, even sum.
    //   Coset 1 (D8+1/2): all coords half-integer, even sum when halved.
    // -----------------------------------------------------------------------
    #[test]
    fn decoded_points_are_valid_e8() {
        let mut state = 0xc0de_cafe_1234_5678u64;
        for &n in &[2u32, 3u32] {
            let q = if n == 3 {
                QUANT_STEP_MFP3
            } else {
                QUANT_STEP_MFP2
            };
            for _ in 0..200_000 {
                let mut v = [0.0f32; 8];
                for i in 0..8 {
                    v[i] = lcg_next(&mut state) * 6.0;
                }
                let idx = quantize8_n(&v, q, n);
                let p = decode_index_n(idx, n);

                let coset = (idx >> (8 * n - 1)) & 1;
                if coset == 0 {
                    // D8: all coords integer.
                    for i in 0..8 {
                        assert!(
                            p[i].fract().abs() < 1e-6,
                            "n={n} D8 non-integer coord[{i}]={:.4}",
                            p[i]
                        );
                    }
                    // Even sum.
                    let s: i32 = p.iter().map(|&x| x as i32).sum();
                    assert_eq!(s & 1, 0, "n={n} D8 odd integer sum: {s}, p={p:?}");
                } else {
                    // D8+1/2: all coords half-integer (x - 0.5 is integer).
                    for i in 0..8 {
                        assert!(
                            (p[i].fract().abs() - 0.5).abs() < 1e-5,
                            "n={n} coset non-half-integer coord[{i}]={:.4}",
                            p[i]
                        );
                    }
                    // Even sum when shifted: sum(p[i]-0.5) = sum(p[i]) - 4 must be even.
                    let s_shifted: f32 = p.iter().map(|&x| x - 0.5).sum::<f32>();
                    let s_int = s_shifted.round() as i32;
                    assert_eq!(
                        s_int & 1,
                        0,
                        "n={n} coset shifted-sum odd: {s_int}, p={p:?}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test C — host-mirror: pure-Rust transliteration of the kernel C decode
    //           logic matches decode_index_n for all codewords in range.
    //           (Exhaustive for n=2: 2^16 = 65536; sampled for n=3: 16M.)
    // -----------------------------------------------------------------------

    /// Exact transliteration of the kernel's mfp3_decode_index / mfp2_decode_index C code.
    /// This mirrors B's device helpers so any packing divergence between Rust and HIP
    /// fails the test rather than silently producing NaN on the GPU.
    fn kernel_mirror_decode(idx: u32, n: u32) -> [f32; 8] {
        match n {
            3 => {
                // mfp3_decode_index: 3-bit nibbles, center 4, coset bit 23, e7_high 2b @ 21.
                let coset = (idx >> 23) & 1;
                let mut e = [0u32; 8];
                let mut sl: u32 = 0;
                for i in 0..7_u32 {
                    e[i as usize] = (idx >> (3 * i)) & 0x7;
                    sl += e[i as usize];
                }
                let e7_high = (idx >> 21) & 0x3;
                let p7 = e7_high << 1;
                e[7] = p7 | ((sl + p7) & 1);
                let mut p = [0.0f32; 8];
                for i in 0..8 {
                    let c = (e[i] as i32 - 4) as f32;
                    p[i] = if coset == 1 { c + 0.5 } else { c };
                }
                p
            }
            2 => {
                // mfp2_decode_index: 2-bit nibbles, center 2, coset bit 15, e7_high 1b @ 14.
                let coset = (idx >> 15) & 1;
                let mut e = [0u32; 8];
                let mut sl: u32 = 0;
                for i in 0..7_u32 {
                    e[i as usize] = (idx >> (2 * i)) & 0x3;
                    sl += e[i as usize];
                }
                let e7_high = (idx >> 14) & 0x1;
                let p7 = e7_high << 1;
                e[7] = p7 | ((sl + p7) & 1);
                let mut p = [0.0f32; 8];
                for i in 0..8 {
                    let c = (e[i] as i32 - 2) as f32;
                    p[i] = if coset == 1 { c + 0.5 } else { c };
                }
                p
            }
            _ => panic!("kernel_mirror_decode: unsupported n={n}"),
        }
    }

    #[test]
    fn host_kernel_mirror_n2_exhaustive() {
        // n=2: 2^16 = 65536 codewords, exhaustive.
        for idx in 0u32..(1 << 16) {
            let rust = decode_index_n(idx, 2);
            let kern = kernel_mirror_decode(idx, 2);
            for i in 0..8 {
                assert!(
                    (rust[i] - kern[i]).abs() < 1e-7,
                    "n=2 idx=0x{idx:04x} coord[{i}]: Rust={:.4} kernel={:.4}",
                    rust[i],
                    kern[i]
                );
            }
        }
    }

    #[test]
    fn host_kernel_mirror_n3_sampled() {
        // n=3: 2^24 = 16.7M; sample 1M + edges.
        let mut state = 0x9876_5432_1fed_cba0u64;
        let total = 1_000_000u32;
        for _ in 0..total {
            // Use LCG to generate random 24-bit codewords.
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (state >> 40) as u32 & 0xFF_FFFF; // 24 bits
            let rust = decode_index_n(idx, 3);
            let kern = kernel_mirror_decode(idx, 3);
            for i in 0..8 {
                assert!(
                    (rust[i] - kern[i]).abs() < 1e-7,
                    "n=3 idx=0x{idx:06x} coord[{i}]: Rust={:.4} kernel={:.4}",
                    rust[i],
                    kern[i]
                );
            }
        }
        // Edge cases: all-zeros, all-ones (max valid bits), coset-only.
        for &idx in &[0u32, 0xFF_FFFF, 0x80_0000, 0x7F_FFFF] {
            let rust = decode_index_n(idx, 3);
            let kern = kernel_mirror_decode(idx, 3);
            for i in 0..8 {
                assert!(
                    (rust[i] - kern[i]).abs() < 1e-7,
                    "n=3 edge idx=0x{idx:06x} coord[{i}]: Rust={:.4} kernel={:.4}",
                    rust[i],
                    kern[i]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test E — packing-gain + QUANT_STEP sweep (mfp3 and mfp2).
    //   Sweeps q over candidate values, picks the MSE-minimizing q,
    //   and ASSERTS the locked constants are within 5% of optimal
    //   (so a bad constant change fails loudly).
    //   Also instruments clamp-saturation rate (warns if > 5% for n=3).
    // -----------------------------------------------------------------------
    #[test]
    fn quant_step_sweep_mfp3() {
        // Sweep candidates (interior-bracket the locked optimum at bias=4 so the
        // winner is NOT an edge value). QUANT_STEP_MFP3 must be within 5% of the winner.
        let candidates = [1.6f32, 1.7, 1.8, 1.9, 2.0];
        let n = 3u32;
        let n_blocks = 62_500usize; // 500k weights / 8

        let mut state = 0x1111_2222_3333_4444u64;
        // Pre-generate blocks so all candidates see the same data.
        let mut blocks: Vec<[f32; 8]> = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let mut blk = [0.0f32; 8];
            for i in 0..4 {
                let (a, b) = box_muller(&mut state, E8_SIGMA);
                blk[2 * i] = a;
                blk[2 * i + 1] = b;
            }
            // Block-normalize (same as quantizer: max/6).
            let bmax = blk.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
            if bmax > 0.0 {
                for x in blk.iter_mut() {
                    *x /= bmax / 6.0;
                }
            }
            blocks.push(blk);
        }

        let mut best_mse = f64::MAX;
        let mut best_q = candidates[0];
        for &q in &candidates {
            let mut mse = 0.0f64;
            let mut sat = 0usize;
            let bias = coord_bias_n(n);
            let cmax = coord_max_n(n);
            for blk in &blocks {
                let idx = quantize8_n(blk, q, n);
                let dq = dequantize8_n(idx, q, n);
                for i in 0..8 {
                    let e = blk[i] - dq[i];
                    mse += (e * e) as f64;
                }
                // Saturation check: count coords clamped to [0, cmax].
                let p = decode_index_n(idx, n);
                for i in 0..8 {
                    let raw = if (p[i].fract().abs() - 0.5).abs() < 0.1 {
                        (p[i] - 0.5).round() as i32
                    } else {
                        p[i].round() as i32
                    };
                    let biased = raw + bias;
                    if biased <= 0 || biased >= cmax {
                        sat += 1;
                    }
                }
            }
            let avg_mse = mse / (n_blocks * 8) as f64;
            let sat_rate = sat as f64 / (n_blocks * 8) as f64;
            eprintln!(
                "mfp3 q={q:.3}: mse={avg_mse:.6} sat={:.2}%",
                sat_rate * 100.0
            );
            if avg_mse < best_mse {
                best_mse = avg_mse;
                best_q = q;
            }
        }
        eprintln!("mfp3 best q={best_q:.3} mse={best_mse:.6}");

        // The locked constant must be within 5% of the optimal candidate.
        let ratio = if QUANT_STEP_MFP3 > best_q {
            QUANT_STEP_MFP3 / best_q
        } else {
            best_q / QUANT_STEP_MFP3
        };
        assert!(
            ratio <= 1.05,
            "QUANT_STEP_MFP3={QUANT_STEP_MFP3} deviates >5% from optimal q={best_q} \
             (ratio={ratio:.3}). Retune the constant and the kernel #define."
        );
    }

    #[test]
    fn quant_step_sweep_mfp2() {
        // Interior-bracket the locked optimum at bias=2 (winner must not be an edge).
        let candidates = [3.5f32, 3.7, 3.8, 3.9, 4.1];
        let n = 2u32;
        let n_blocks = 62_500usize;

        let mut state = 0x5555_6666_7777_8888u64;
        let mut blocks: Vec<[f32; 8]> = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let mut blk = [0.0f32; 8];
            for i in 0..4 {
                let (a, b) = box_muller(&mut state, E8_SIGMA);
                blk[2 * i] = a;
                blk[2 * i + 1] = b;
            }
            let bmax = blk.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
            if bmax > 0.0 {
                for x in blk.iter_mut() {
                    *x /= bmax / 6.0;
                }
            }
            blocks.push(blk);
        }

        let mut best_mse = f64::MAX;
        let mut best_q = candidates[0];
        for &q in &candidates {
            let mut mse = 0.0f64;
            for blk in &blocks {
                let idx = quantize8_n(blk, q, n);
                let dq = dequantize8_n(idx, q, n);
                for i in 0..8 {
                    let e = blk[i] - dq[i];
                    mse += (e * e) as f64;
                }
            }
            let avg_mse = mse / (n_blocks * 8) as f64;
            eprintln!("mfp2 q={q:.3}: mse={avg_mse:.6}");
            if avg_mse < best_mse {
                best_mse = avg_mse;
                best_q = q;
            }
        }
        eprintln!("mfp2 best q={best_q:.3} mse={best_mse:.6}");

        let ratio = if QUANT_STEP_MFP2 > best_q {
            QUANT_STEP_MFP2 / best_q
        } else {
            best_q / QUANT_STEP_MFP2
        };
        assert!(
            ratio <= 1.05,
            "QUANT_STEP_MFP2={QUANT_STEP_MFP2} deviates >5% from optimal q={best_q} \
             (ratio={ratio:.3}). Retune the constant and the kernel #define."
        );
    }

    // ----------------------------------------------------------------------
    // FINALIZE: explicit CPU-only encode->decode roundtrip PARITY test for
    // mfp3-E8 (n=3) and mfp2-E8 (n=2). Asserts:
    //   (a) decode produces no NaN/Inf on random + Gaussian weight blocks,
    //   (b) reconstruction error is bounded by the lattice cell size for
    //       in-box vectors (rigorous E8 covering-radius bound: ||err||2 <= q),
    //   (c) the round-trip is deterministic (same input -> identical codeword
    //       and identical decoded f32 bytes across repeated calls).
    // Also reports the empirical RMS error on the FWHT-Gaussian (sigma=1.7)
    // for both n, as informational context for the quality A/B downstream.
    // ----------------------------------------------------------------------

    /// E8 covering radius (Conway-Sloane normalization, D8 min-norm 2). For any
    /// u in R^8, ||u - closest_e8(u)||2 <= 1.0. After scaling by q, the L2
    /// reconstruction error of an UNCLAMPED quantization is <= q * R_COV.
    const E8_COVERING_RADIUS: f32 = 1.0;

    /// Build a random E8 lattice point whose biased coords are strictly INSIDE
    /// the n-bit box (in [1, 2^n-2]) so that encode never clamp-saturates and
    /// the covering-radius bound is rigorous. Returns the f32[8] lattice point.
    fn random_inbox_e8_point(state: &mut u64, n: u32) -> [f32; 8] {
        let bias = coord_bias_n(n);
        let cmax = coord_max_n(n);
        // Interior biased range [1, cmax-1] guarantees +-1 parity nudge stays in box.
        let lo = 1i32;
        let hi = cmax - 1; // n=3 -> [1,6], n=2 -> [1,2]
        let span = (hi - lo + 1).max(1) as u32;
        let coset = (((*state >> 17) & 1) as u32) & if n >= 2 { 1 } else { 0 };
        let mut e = [0i32; 8];
        let mut s = 0i32;
        for i in 0..8 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((*state >> 33) as u32) % span;
            e[i] = lo + r as i32;
            s += e[i];
        }
        // Force even sum (E8/D8 constraint) by nudging e[7] inside the interior box.
        if (s & 1) != 0 {
            if e[7] < hi {
                e[7] += 1;
            } else {
                e[7] -= 1;
            }
        }
        let mut p = [0.0f32; 8];
        for i in 0..8 {
            let c = (e[i] - bias) as f32;
            p[i] = if coset == 1 { c + 0.5 } else { c };
        }
        let _ = cmax;
        p
    }

    fn roundtrip_parity_for_n(n: u32, q: f32) {
        // ---- (a) NaN/Inf-free + (c) determinism on random & Gaussian blocks ----
        let mut state = 0xDEAD_BEEF_1234_5678u64 ^ ((n as u64) << 40);
        let n_random = 200_000usize;
        for _ in 0..n_random {
            // Mix of uniform-random (full dynamic range, exercises clamp) and
            // FWHT-Gaussian-like blocks.
            let mut v = [0.0f32; 8];
            for i in 0..4 {
                let (a, b) = box_muller(&mut state, E8_SIGMA);
                // scale to roughly the block-normalized [-6,6] domain
                v[2 * i] = a;
                v[2 * i + 1] = b;
            }
            // occasionally push values hard to the rails to exercise saturation
            if (state & 0x7) == 0 {
                for x in v.iter_mut() {
                    *x *= 4.0;
                }
            }

            let idx1 = quantize8_n(&v, q, n);
            let dq1 = dequantize8_n(idx1, q, n);

            // (a) no NaN/Inf anywhere
            for i in 0..8 {
                assert!(
                    dq1[i].is_finite(),
                    "n={n}: non-finite decode dq[{i}]={} for v={v:?} idx=0x{idx1:08x}",
                    dq1[i]
                );
            }
            // codeword occupies only the low 8n bits (narrow-read safety)
            assert_eq!(
                idx1 >> (8 * n),
                0,
                "n={n}: codeword 0x{idx1:08x} has bits above 8n={}",
                8 * n
            );

            // (c) determinism: same input -> identical codeword AND identical bytes
            let idx2 = quantize8_n(&v, q, n);
            assert_eq!(idx1, idx2, "n={n}: non-deterministic encode for v={v:?}");
            let dq2 = dequantize8_n(idx2, q, n);
            for i in 0..8 {
                assert_eq!(
                    dq1[i].to_bits(),
                    dq2[i].to_bits(),
                    "n={n}: non-deterministic decode dq[{i}] {} vs {}",
                    dq1[i],
                    dq2[i]
                );
            }
        }

        // ---- (b) error bounded by lattice cell size on guaranteed in-box points ----
        // For an in-box vector u (lattice point + jitter < min-dist/2), closest_e8
        // returns the same lattice point, so dequant == q * point exactly modulo the
        // small jitter, and ||v - dequant||2 <= q * R_cov rigorously.
        let mut bstate = 0x0BAD_F00D_CAFE_2026u64 ^ ((n as u64) << 32);
        let n_box = 100_000usize;
        let bound = q * E8_COVERING_RADIUS;
        let mut max_err = 0.0f32;
        for _ in 0..n_box {
            let point = random_inbox_e8_point(&mut bstate, n); // an actual E8 point
                                                               // Build v = q * point + jitter, jitter small enough to stay in the
                                                               // Voronoi cell of `point` (E8 packing radius = sqrt(2)/2 in lattice
                                                               // units; use a conservative 0.30 to stay well inside).
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                bstate = bstate
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (((bstate >> 40) as f32) / (u32::MAX as f32)) * 2.0 - 1.0; // [-1,1]
                v[i] = q * point[i] + 0.30 * q * j;
            }
            let idx = quantize8_n(&v, q, n);
            let dq = dequantize8_n(idx, q, n);
            let mut sq = 0.0f32;
            for i in 0..8 {
                assert!(dq[i].is_finite(), "n={n}: non-finite in-box decode");
                let e = v[i] - dq[i];
                sq += e * e;
            }
            let l2 = sq.sqrt();
            if l2 > max_err {
                max_err = l2;
            }
            // Allow a small epsilon for fp32 rounding in the closest-point search.
            assert!(
                l2 <= bound + 1e-3,
                "n={n}: in-box L2 recon error {l2:.5} exceeds lattice cell bound \
                 q*R_cov={bound:.5} (q={q})"
            );
        }

        // ---- empirical RMS on the FWHT-Gaussian (informational) ----
        let mut gstate = 0xFEED_FACE_5151_3030u64 ^ ((n as u64) << 24);
        let n_g = 200_000usize;
        let mut mse = 0.0f64;
        for _ in 0..n_g {
            let mut blk = [0.0f32; 8];
            for i in 0..4 {
                let (a, b) = box_muller(&mut gstate, E8_SIGMA);
                blk[2 * i] = a;
                blk[2 * i + 1] = b;
            }
            let bmax = blk.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
            if bmax > 0.0 {
                for x in blk.iter_mut() {
                    *x /= bmax / 6.0;
                }
            }
            let dq = dequantize8_n(quantize8_n(&blk, q, n), q, n);
            for i in 0..8 {
                let e = (blk[i] - dq[i]) as f64;
                mse += e * e;
            }
        }
        let rms = (mse / (n_g * 8) as f64).sqrt();
        eprintln!(
            "roundtrip n={n} q={q}: in-box max L2 err {max_err:.5} (bound {bound:.5}); \
             FWHT-Gaussian RMS {rms:.5} over {n_g} blocks"
        );
    }

    #[test]
    fn roundtrip_parity_mfp3() {
        roundtrip_parity_for_n(3, QUANT_STEP_MFP3);
    }

    #[test]
    fn roundtrip_parity_mfp2() {
        roundtrip_parity_for_n(2, QUANT_STEP_MFP2);
    }

    /// Quantization shrinkage: does the E8 round-trip preserve magnitude?
    ///
    /// MSE-optimal quantization of a zero-mean distribution is systematically
    /// *shrinking* — the reconstruction correlates with the source but is
    /// shorter than it, because spending error budget on magnitude is cheaper
    /// than on direction. The retained-energy ratio `E[w_hat.w] / E[w^2]`
    /// quantifies it, and `E[w_hat.w] / E[w_hat^2]` is the per-group rescale
    /// that would undo it.
    ///
    /// This matters beyond curiosity. DeepSeek V4 ships a `route_scale` of 1.8
    /// (mq2r) and 2.2 (other builds) where the checkpoint declares 1.5, and the
    /// PyTorch reference scores PPL 4.693 at 1.5 — so 1.5 is right for the
    /// model and our routed branch is systematically weak. `route_scale`
    /// multiplies *only* the routed-expert branch, which is exactly the tier
    /// quantized through this codec, so shrinkage here is a candidate cause and
    /// a global scalar is at best its average.
    ///
    /// Informational: asserts only that the round-trip is sane (positive
    /// correlation, no blow-up), and prints the factors under `--nocapture`.
    #[test]
    fn e8_shrinkage_on_fwht_gaussian() {
        // xorshift + Box-Muller; no dev-dependency, fully deterministic.
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut gauss = move || {
            let u1 = next().max(1e-12);
            let u2 = next();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        };

        println!("\nE8 shrinkage (QUANT_STEP={QUANT_STEP}, E8_SIGMA={E8_SIGMA})");
        println!(
            "{:<10} {:>10} {:>12} {:>12}",
            "sigma", "retained", "|wh|/|w|", "opt_gain"
        );
        for sigma in [1.0f32, E8_SIGMA, 2.5] {
            let (mut dot, mut ss_w, mut ss_h) = (0.0f64, 0.0f64, 0.0f64);
            for _ in 0..40_000 {
                let mut w = [0.0f32; 8];
                for v in w.iter_mut() {
                    *v = gauss() * sigma;
                }
                let h = dequantize8(quantize8(&w, QUANT_STEP), QUANT_STEP);
                for i in 0..8 {
                    dot += f64::from(h[i]) * f64::from(w[i]);
                    ss_w += f64::from(w[i]) * f64::from(w[i]);
                    ss_h += f64::from(h[i]) * f64::from(h[i]);
                }
            }
            let retained = dot / ss_w;
            let norm_ratio = (ss_h / ss_w).sqrt();
            let opt_gain = dot / ss_h;
            println!("{sigma:<10} {retained:>10.4} {norm_ratio:>12.4} {opt_gain:>12.4}");
            assert!(
                retained > 0.3 && retained < 1.3,
                "sigma={sigma}: retained energy {retained} is not a sane round-trip"
            );
            assert!(
                norm_ratio.is_finite() && norm_ratio > 0.0,
                "sigma={sigma}: degenerate reconstruction"
            );
        }
        println!(
            "opt_gain is the per-group rescale minimising MSE; a global \
             route_scale can only approximate its average.\n"
        );
    }
}
