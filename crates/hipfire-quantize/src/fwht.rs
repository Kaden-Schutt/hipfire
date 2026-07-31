// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fast Walsh-Hadamard transform over 256-element groups, plus the sign tables.
//!
//! This MUST stay bit-identical to `fwht_forward_256` in
//! `kernels/src/turbo_common.h` and to the engine's `gen_fwht_signs`: the
//! quantizer bakes the rotation into the weights while the runtime applies the
//! matching rotation to activations, and `dot(rot(W), rot(x)) == dot(W, x)`
//! only holds if both sides agree exactly. That is why this is hand-rolled
//! rather than delegated to an FFT/transform crate.
//!
//! Order is: `signs1` → butterfly → `1/16` scale → `signs2`. The `1/16` makes
//! the transform orthonormal (`1/sqrt(256)`), so squared error in the rotated
//! domain equals squared error in the weight domain — which is what lets the
//! encoders optimize MSE directly on rotated values.

/// Elements per transform group.
pub const N: usize = 256;

/// Generate a `±1` sign table from `seed`, matching the engine's LCG exactly.
pub fn gen_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// In-place FWHT of one 256-element group.
///
/// Panics unless `x.len() == 256`.
pub fn fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == N, "fwht_256 needs exactly {N} elements");
    for i in 0..N {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < N {
        let mut i = 0;
        while i < N {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/sqrt(256)
    for i in 0..N {
        x[i] *= scale * signs2[i];
    }
}

/// In-place inverse FWHT. The transform is self-inverse up to the sign tables,
/// so this undoes [`fwht_256`] exactly.
pub fn inv_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == N, "inv_fwht_256 needs exactly {N} elements");
    let scale = 0.0625;
    for i in 0..N {
        x[i] *= scale * signs2[i];
    }
    let mut stride = 1;
    while stride < N {
        let mut i = 0;
        while i < N {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    for i in 0..N {
        x[i] *= signs1[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signs() -> (Vec<f32>, Vec<f32>) {
        (gen_signs(0x9E37_79B9, N), gen_signs(0x85EB_CA6B, N))
    }

    fn sample(seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..N)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 8388608.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn signs_are_plus_or_minus_one_and_seed_stable() {
        let a = gen_signs(0x9E37_79B9, N);
        assert!(a.iter().all(|&v| v == 1.0 || v == -1.0));
        assert_eq!(a, gen_signs(0x9E37_79B9, N), "must be deterministic");
        assert_ne!(a, gen_signs(0x85EB_CA6B, N), "different seed, different table");
    }

    /// Orthonormality is the property every encoder depends on: it is what
    /// makes rotated-domain MSE equal weight-domain MSE.
    #[test]
    fn transform_is_orthonormal() {
        let (s1, s2) = signs();
        for seed in 0..8u32 {
            let x = sample(seed * 37 + 1);
            let before: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let mut y = x.clone();
            fwht_256(&mut y, &s1, &s2);
            let after: f64 = y.iter().map(|&v| (v as f64) * (v as f64)).sum();
            assert!(
                (after - before).abs() <= 1e-4 * before,
                "energy changed: {before} -> {after}"
            );
        }
    }

    #[test]
    fn inverse_round_trips() {
        let (s1, s2) = signs();
        let x = sample(7);
        let mut y = x.clone();
        fwht_256(&mut y, &s1, &s2);
        inv_fwht_256(&mut y, &s1, &s2);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    /// A constant vector must concentrate all energy in one coefficient — the
    /// classic Hadamard sanity check, modulo the sign tables.
    #[test]
    fn dc_concentrates() {
        let (s1, s2) = signs();
        let mut x: Vec<f32> = s1.iter().map(|&s| s).collect(); // undo signs1
        fwht_256(&mut x, &s1, &s2);
        let big = x.iter().filter(|v| v.abs() > 1.0).count();
        assert_eq!(big, 1, "expected a single dominant coefficient");
        assert!((x.iter().map(|v| v.abs()).fold(0.0, f32::max) - 16.0).abs() < 1e-3);
    }
}
