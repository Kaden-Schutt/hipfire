// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure arch-agnostic spec-decode numeric primitives.
//!
//! Relocated from `hipfire-arch-qwen35::speculative` (P2b): residual
//! sampling, temperature softmax, and the bit/abs/rel diff stats used by the
//! rollback-parity comparators. All operate on plain slices — no GPU or arch
//! types — so the strategy crates share them directly.

#[derive(Debug)]
pub struct F32DiffStats {
    pub words: usize,
    pub bit_different_words: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub max_rel: f32,
}

#[derive(Debug)]
pub struct LogitDiffStats {
    pub words: usize,
    pub bit_different_words: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub max_rel: f32,
}

pub fn logit_diff_stats(actual: &[f32], expected: &[f32]) -> LogitDiffStats {
    let mut words = 0usize;
    let mut bit_different_words = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    for (&actual_value, &expected_value) in actual.iter().zip(expected.iter()) {
        words += 1;
        if actual_value.to_bits() != expected_value.to_bits() {
            bit_different_words += 1;
        }
        let abs = (actual_value - expected_value).abs();
        if abs.is_finite() {
            max_abs = max_abs.max(abs);
            sum_abs += abs as f64;
            let denom = expected_value.abs().max(1.0e-12);
            max_rel = max_rel.max(abs / denom);
        }
    }
    LogitDiffStats {
        words,
        bit_different_words,
        max_abs,
        mean_abs: if words == 0 {
            0.0
        } else {
            (sum_abs / words as f64) as f32
        },
        max_rel,
    }
}

pub fn first_mismatched_f32(bytes: &[u8], first_offset: usize) -> Option<f32> {
    let float_offset = first_offset.saturating_sub(first_offset % 4);
    let word = bytes.get(float_offset..float_offset + 4)?;
    Some(f32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
}

pub fn f32_diff_stats(actual: &[u8], expected: &[u8]) -> Option<F32DiffStats> {
    if actual.len() != expected.len() || actual.len() % 4 != 0 {
        return None;
    }
    let mut words = 0usize;
    let mut bit_different_words = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    for (actual_word, expected_word) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let actual_value = f32::from_ne_bytes([
            actual_word[0],
            actual_word[1],
            actual_word[2],
            actual_word[3],
        ]);
        let expected_value = f32::from_ne_bytes([
            expected_word[0],
            expected_word[1],
            expected_word[2],
            expected_word[3],
        ]);
        words += 1;
        if actual_value.to_bits() != expected_value.to_bits() {
            bit_different_words += 1;
        }
        let abs = (actual_value - expected_value).abs();
        if abs.is_finite() {
            max_abs = max_abs.max(abs);
            sum_abs += abs as f64;
            let denom = expected_value.abs().max(1.0e-12);
            max_rel = max_rel.max(abs / denom);
        }
    }
    Some(F32DiffStats {
        words,
        bit_different_words,
        max_abs,
        mean_abs: if words == 0 {
            0.0
        } else {
            (sum_abs / words as f64) as f32
        },
        max_rel,
    })
}

/// Single-pass argmax for token sampling. Not SIMD-optimized — the logit
/// vector is downloaded once per verify step so the CPU scan cost is
/// negligible relative to GEMV work.
#[inline]
pub fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature-scaled softmax. Writes into `out` (reused across calls to
/// avoid per-position allocation in the rejection-sampling hot loop).
#[inline]
pub fn softmax_temp_into(logits: &[f32], temp: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(logits.len());
    let inv_t = 1.0 / temp;
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let s = v * inv_t;
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v * inv_t - max).exp();
        out.push(e);
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    for p in out.iter_mut() {
        *p *= inv_sum;
    }
}

/// Draw from (p_target − p_draft)₊, renormalized. Used on rejection to
/// sample the "corrective" bonus token in speculative rejection sampling
/// (Chen & Leviathan 2023, algorithm 1).
#[inline]
pub fn sample_residual(p_target: &[f32], p_draft: &[f32], u: f32) -> u32 {
    let mut sum = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 0.0 {
        // Degenerate case (p_draft >= p_target everywhere). Should not
        // happen in practice if a rejection was just drawn. Fall back to
        // argmax of p_target.
        return argmax_u32(p_target);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 {
            acc += d;
            if u_scaled < acc {
                return i as u32;
            }
        }
    }
    (p_target.len() - 1) as u32
}
