// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validate the GPU QTIP trellis encoder (`qtip_viterbi_encode`, full Viterbi)
//! against the CPU beam encoder on random Gaussian groups. The GPU path is exact
//! Viterbi (optimal), so its reconstruction MSE must be ≤ the CPU beam MSE. The
//! trellis codebook + decode/scale helpers are replicated from
//! `hipfire_quantize::qtip` so this stays a self-contained hipfire-rdna example.
//!
//!   cargo run --release -p hipfire-rdna --example parity_qtip_viterbi_encode [n_groups bits beam]

use hipfire_rdna::Gpu;

const STATE_BITS: u32 = 12;
const NUM_STATES: usize = 1 << STATE_BITS;
const STATE_MASK: u32 = (NUM_STATES as u32) - 1;

fn decode_1mad(state: u32) -> f32 {
    let x = (state as u64) & 0xFFFF_FFFF;
    let x = x.wrapping_mul(34_038_481).wrapping_add(76_625_530) & 0xFFFF_FFFF;
    let bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF);
    (bs as f32 - 510.0) / 147.800_537_109_375
}
fn build_codebook() -> Vec<f32> {
    let mut cb: Vec<f64> = (0..NUM_STATES as u32)
        .map(|s| decode_1mad(s) as f64)
        .collect();
    let mean = cb.iter().sum::<f64>() / cb.len() as f64;
    for v in cb.iter_mut() {
        *v -= mean;
    }
    let var = cb.iter().map(|v| v * v).sum::<f64>() / cb.len() as f64;
    let inv = if var > 0.0 { 1.0 / var.sqrt() } else { 1.0 };
    cb.iter().map(|v| (v * inv) as f32).collect()
}
fn decode_group_bits(symbols: &[u8], scale: f32, cb: &[f32], bits: u32) -> Vec<f32> {
    let sym_mask = (1u32 << bits) - 1;
    let mut state: u32 = 0;
    symbols
        .iter()
        .map(|&sym| {
            state = ((state << bits) | (sym as u32 & sym_mask)) & STATE_MASK;
            scale * cb[state as usize]
        })
        .collect()
}
fn group_scale(w: &[f32]) -> f32 {
    let ss: f64 = w.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (ss / w.len() as f64).sqrt() as f32
}
fn optimal_scale_bits(w: &[f32], symbols: &[u8], cb: &[f32], bits: u32) -> f32 {
    let sym_mask = (1u32 << bits) - 1;
    let mut state: u32 = 0;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &sym) in symbols.iter().enumerate() {
        state = ((state << bits) | (sym as u32 & sym_mask)) & STATE_MASK;
        let c = cb[state as usize] as f64;
        num += w[i] as f64 * c;
        den += c * c;
    }
    if den > 0.0 {
        (num / den) as f32
    } else {
        group_scale(w)
    }
}
fn beam_encode_group_bits(
    w: &[f32],
    scale: f32,
    cb: &[f32],
    beam_width: usize,
    bits: u32,
) -> Vec<u8> {
    let num_symbols = 1usize << bits;
    let n = w.len();
    let mut beam: Vec<(u32, f64)> = vec![(0u32, 0.0)];
    let mut steps: Vec<Vec<(u32, u32, u8)>> = Vec::with_capacity(n);
    let mut cand: Vec<(u32, f64, u32, u8)> = Vec::with_capacity(beam_width * num_symbols);
    for &wi in w {
        let wi = wi as f64;
        cand.clear();
        for (bi, &(s_prev, c_prev)) in beam.iter().enumerate() {
            let base = (s_prev << bits) & STATE_MASK;
            for sym in 0..num_symbols as u32 {
                let s_new = base | sym;
                let diff = wi - scale as f64 * cb[s_new as usize] as f64;
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

/// Deterministic standard-normal samples (Box–Muller over an LCG).
fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    let mut u01 = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-12, 1.0)
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (u01(), u01());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| ((x - y) as f64) * ((x - y) as f64))
        .sum::<f64>()
        / a.len() as f64
}

fn main() {
    let mut a = std::env::args().skip(1);
    let n_groups: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let bits: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let beam: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let cb = build_codebook();

    let mut gpu = Gpu::init().unwrap();

    // Random Gaussian groups (post-rotation weight statistics).
    let w_all = gaussian(0x51A7, n_groups * 256);
    let wd = gpu
        .upload_raw(
            &w_all
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[n_groups, 256],
        )
        .unwrap();
    let symbols = gpu
        .upload_raw(&vec![0u8; n_groups * 256], &[n_groups, 256])
        .unwrap();
    // Packed backpointer: [n_groups][256 pos][256 threads][2 u32] = n_groups*512 KB.
    let backptr = gpu
        .upload_raw(
            &vec![0u8; n_groups * 256 * 256 * 2 * 4],
            &[n_groups * 256 * 256 * 2],
        )
        .unwrap();
    let scales = gpu
        .upload_raw(&vec![0u8; n_groups * 4], &[n_groups])
        .unwrap();

    let t0 = std::time::Instant::now();
    gpu.qtip_viterbi_encode(&wd, &symbols, &backptr, &scales, n_groups, bits)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    let gsym = gpu.download_raw(&symbols, n_groups * 256).unwrap();
    let gsc = gpu.download_f32(&scales).unwrap();

    // Per-group reconstruction MSE: GPU Viterbi vs CPU beam. Both re-fit the
    // closed-form optimal scale before decoding (as the production pipeline does).
    let (mut sum_gpu, mut sum_cpu, mut worst_ratio) = (0.0f64, 0.0f64, 0.0f64);
    let mut scale_mismatch = 0.0f32;
    let t1 = std::time::Instant::now();
    for g in 0..n_groups {
        let w = &w_all[g * 256..g * 256 + 256];
        // GPU symbols
        let gs = &gsym[g * 256..g * 256 + 256];
        let sopt = optimal_scale_bits(w, gs, &cb, bits);
        let m_gpu = mse(w, &decode_group_bits(gs, sopt, &cb, bits));
        // GPU-computed RMS scale should match the CPU group_scale seed.
        scale_mismatch = scale_mismatch.max((gsc[g] - group_scale(w)).abs());
        // CPU beam reference (same RMS seed)
        let cs = beam_encode_group_bits(w, group_scale(w), &cb, beam, bits);
        let copt = optimal_scale_bits(w, &cs, &cb, bits);
        let m_cpu = mse(w, &decode_group_bits(&cs, copt, &cb, bits));
        sum_gpu += m_gpu;
        sum_cpu += m_cpu;
        if m_cpu > 0.0 {
            worst_ratio = worst_ratio.max(m_gpu / m_cpu);
        }
    }
    let cpu_ms = t1.elapsed().as_secs_f64() * 1e3;
    let (mgpu, mcpu) = (sum_gpu / n_groups as f64, sum_cpu / n_groups as f64);
    // Viterbi is optimal, so mean GPU MSE must be ≤ beam MSE (tiny slack for f32).
    let pass = mgpu <= mcpu * 1.02 && scale_mismatch < 1e-4;
    println!(
        "parity_qtip_viterbi_encode n_groups={n_groups} bits={bits} beam={beam} on {}:\n  \
         GPU-Viterbi MSE={mgpu:.6}  CPU-beam MSE={mcpu:.6}  mean_ratio={:.4}  worst_group_ratio={worst_ratio:.4}\n  \
         scale_mismatch={scale_mismatch:.2e}  GPU_encode={gpu_ms:.1}ms  CPU_beam={cpu_ms:.1}ms  -> {}",
        gpu.arch,
        mgpu / mcpu,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
