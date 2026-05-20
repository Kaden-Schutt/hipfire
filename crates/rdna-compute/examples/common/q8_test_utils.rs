// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Shared test utilities for Q8_0 unit tests and microbenches.
//
// `cargo` examples don't share a crate-level test-utils module the way unit
// tests can; each example is a separate top-level binary. We include this
// file from each consumer via `#[path = "common/q8_test_utils.rs"] mod ...`.
//
// Keep the surface minimal — these are pure helpers, no GPU state.

#![allow(dead_code)] // each include site uses a subset; suppress per-binary warnings.

/// IEEE-754 float32 → float16 bit pattern (round-to-zero on mantissa).
///
/// Used to construct fp16 scale fields for synthetic Q8_0 weight tensors.
/// Handles the cases we actually exercise: normal positive/negative numbers
/// well within fp16 range. Saturates to +/- inf above the fp16 max-normal
/// exponent and flushes subnormals to zero.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp_f32 == 0 {
        return sign;
    }
    if exp_f32 == 0xff {
        return sign | 0x7c00 | if mant != 0 { 1 } else { 0 };
    }
    let exp = exp_f32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Synthesize a Q8_0 weight tensor of shape [m, k] with deterministic PRNG.
///
/// Layout: row-major. Each row has `k / 32` blocks of 34 bytes:
///   `[fp16 scale (2 B) | int8[32] weights (32 B)]`
/// Scales are in [0.001, 0.050]; weight bytes are in [-127, 127].
///
/// `k` must be a multiple of 32.
pub fn synth_q8(m: usize, k: usize, seed0: u32) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; m * bpr * 34];
    let mut seed = seed0;
    let mut prng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };
    for r in 0..m {
        for b in 0..bpr {
            let off = r * bpr * 34 + b * 34;
            let sf = 0.001 + (prng() as f32 / u32::MAX as f32) * 0.049;
            let sb = f32_to_f16_bits(sf);
            out[off] = (sb & 0xFF) as u8;
            out[off + 1] = (sb >> 8) as u8;
            for j in 0..32 {
                out[off + 2 + j] = ((prng() as i32 % 255) - 127) as i8 as u8;
            }
        }
    }
    out
}
