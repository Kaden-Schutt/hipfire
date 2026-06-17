// SPDX-License-Identifier: Apache-2.0
//! QTIP quantize→dequant to fp32, for building a quantized (frozen) training
//! base. Phase 2 (docs/plans/2026-06-17-hipfire-train-phase0.md → Phase 2).
//!
//! The encoder primitives (bitshift-trellis 1MAD codebook, beam Viterbi encode,
//! decode, scales) and the FWHT incoherence rotation are **vendored** from
//! `crates/hipfire-quantize/src/qtip.rs` + its crate-root FWHT helpers, because
//! that crate is bin-only (no lib target) and extracting one is a large refactor
//! of an 11.5k-line file. De-dup later by giving hipfire-quantize a lib target.
//!
//! We decode QTIP back to fp32 once (the codes never change in recovery FT), so
//! the training forward stays the verified fp32 path — no GPU decode kernel
//! needed. `cpu_fwht_256` is orthogonal ((1/16)²·H² = I, signs involutive), so
//! the inverse rotation is the same routine with the sign vectors swapped.

const STATE_BITS: u32 = 12;
const NUM_STATES: usize = 1 << STATE_BITS;
const STATE_MASK: u32 = (NUM_STATES as u32) - 1;
const GROUP: usize = 256;

#[inline]
fn decode_1mad(state: u32) -> f32 {
    let x = (state as u64) & 0xFFFF_FFFF;
    let x = x.wrapping_mul(34_038_481).wrapping_add(76_625_530) & 0xFFFF_FFFF;
    let byte_sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF);
    (byte_sum as f32 - 510.0) / 147.800_537_109_375
}

pub fn build_codebook() -> Vec<f32> {
    let mut cb: Vec<f64> = (0..NUM_STATES as u32).map(|s| decode_1mad(s) as f64).collect();
    let mean = cb.iter().sum::<f64>() / cb.len() as f64;
    for v in cb.iter_mut() {
        *v -= mean;
    }
    let var = cb.iter().map(|v| v * v).sum::<f64>() / cb.len() as f64;
    let inv_std = if var > 0.0 { 1.0 / var.sqrt() } else { 1.0 };
    cb.iter().map(|v| (v * inv_std) as f32).collect()
}

fn decode_group_bits(symbols: &[u8], scale: f32, codebook: &[f32], bits: u32) -> Vec<f32> {
    let sym_mask = (1u32 << bits) - 1;
    let mut state: u32 = 0;
    let mut out = Vec::with_capacity(symbols.len());
    for &sym in symbols {
        state = ((state << bits) | (sym as u32 & sym_mask)) & STATE_MASK;
        out.push(scale * codebook[state as usize]);
    }
    out
}

fn beam_encode_group_bits(
    weights: &[f32],
    scale: f32,
    codebook: &[f32],
    beam_width: usize,
    bits: u32,
) -> Vec<u8> {
    let num_symbols = 1usize << bits;
    let n = weights.len();
    let mut beam: Vec<(u32, f64)> = vec![(0u32, 0.0)];
    let mut steps: Vec<Vec<(u32, u32, u8)>> = Vec::with_capacity(n);
    let mut cand: Vec<(u32, f64, u32, u8)> = Vec::with_capacity(beam_width * num_symbols);

    for &w in weights {
        let w = w as f64;
        cand.clear();
        for (bi, &(s_prev, c_prev)) in beam.iter().enumerate() {
            let base = (s_prev << bits) & STATE_MASK;
            for sym in 0..num_symbols as u32 {
                let s_new = base | sym;
                let diff = w - scale as f64 * codebook[s_new as usize] as f64;
                cand.push((s_new, c_prev + diff * diff, bi as u32, sym as u8));
            }
        }
        cand.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        cand.dedup_by_key(|c| c.0);
        if cand.len() > beam_width {
            cand.select_nth_unstable_by(beam_width, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(beam_width);
        }
        let mut rec = Vec::with_capacity(cand.len());
        let mut next_beam = Vec::with_capacity(cand.len());
        for &(st, c, pi, sy) in cand.iter() {
            rec.push((st, pi, sy));
            next_beam.push((st, c));
        }
        steps.push(rec);
        beam = next_beam;
    }

    let mut best_idx = 0usize;
    let mut best_cost = f64::INFINITY;
    for (i, &(_, c)) in beam.iter().enumerate() {
        if c < best_cost {
            best_cost = c;
            best_idx = i;
        }
    }
    let mut symbols = vec![0u8; n];
    let mut idx = best_idx;
    for step in (0..n).rev() {
        let (_, prev_idx, sym) = steps[step][idx];
        symbols[step] = sym;
        idx = prev_idx as usize;
    }
    symbols
}

fn group_scale(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 1.0;
    }
    let ss: f64 = weights.iter().map(|&w| (w as f64) * (w as f64)).sum();
    (ss / weights.len() as f64).sqrt() as f32
}

fn optimal_scale_bits(weights: &[f32], symbols: &[u8], codebook: &[f32], bits: u32) -> f32 {
    let sym_mask = (1u32 << bits) - 1;
    let mut state: u32 = 0;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &sym) in symbols.iter().enumerate() {
        state = ((state << bits) | (sym as u32 & sym_mask)) & STATE_MASK;
        let c = codebook[state as usize] as f64;
        num += weights[i] as f64 * c;
        den += c * c;
    }
    if den > 0.0 {
        (num / den) as f32
    } else {
        group_scale(weights)
    }
}

/// Deterministic ±1 sign vector (LCG), matching hipfire-quantize.
pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
        })
        .collect()
}

/// FWHT-256 with sign pre/post-multiply: `out = signs2 ∘ (1/16·H·(signs1∘x))`.
/// Orthogonal; inverse = same call with signs1/signs2 swapped.
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
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
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}

/// QTIP quantize→dequant of a flat row-major weight buffer (len % 256 == 0).
/// Per 256-group: FWHT-rotate → beam-encode (`bits`-bit trellis) → decode →
/// inverse-FWHT. Returns the fp32 dequantized weights (`hatW`) in weight space.
pub fn qtip_quantize_dequant(w: &[f32], bits: u32, beam_width: usize) -> Vec<f32> {
    assert!(w.len() % GROUP == 0, "weight len {} not a multiple of 256", w.len());
    let cb = build_codebook();
    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    let mut out = vec![0.0f32; w.len()];
    for b in 0..w.len() / GROUP {
        let mut g = [0.0f32; GROUP];
        g.copy_from_slice(&w[b * GROUP..(b + 1) * GROUP]);
        cpu_fwht_256(&mut g, &s1, &s2); // rotate into ≈Gaussian space
        let s0 = group_scale(&g);
        let sym = beam_encode_group_bits(&g, s0, &cb, beam_width, bits);
        let s = optimal_scale_bits(&g, &sym, &cb, bits);
        let mut hat = decode_group_bits(&sym, s, &cb, bits);
        cpu_fwht_256(&mut hat, &s2, &s1); // inverse rotation (signs swapped)
        out[b * GROUP..(b + 1) * GROUP].copy_from_slice(&hat);
    }
    out
}
