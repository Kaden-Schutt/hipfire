// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.


#![allow(dead_code, unused_imports, unused_variables, non_snake_case, clippy::all)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use hipfire_quantize::hessian_io;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::reap_overlay;
use crate::quant_mq::*;
use crate::quant_hfp4::*;
use crate::quant_fwht::{cpu_fwht_256, gen_fwht_signs};
use crate::dequant::{dequantize_e2m1_ue8m0_to_f32, e2m1_to_f32};
use crate::calibration::awq_eligible;
use crate::quant_e8::*;
use crate::hfq::*;

mod gptq_damping_probe {
    //! Offline GPTQ-Lloyd damping sweep. Runs the GPTQ-Lloyd quant pipeline
    //! against synthetic DeepSeek V4-realistic weight distributions across a damping
    //! range, compares per-block reconstruction MSE to plain Lloyd. Catches
    //! a known failure mode where forward-error-propagation on FWHT-rotated
    //! (largely-decorrelated) weights INJECTS noise rather than removing it
    //! at moderate-to-high damping values — what the DeepSeek V4 MQ2-GPTQ-all run
    //! is suspected to be hitting.
    //!
    //! Run with:
    //!   cargo test -p hipfire-quantize gptq_damping_probe -- --nocapture
    use super::*;
    use crate::quant_fwht::gen_fwht_signs;
    use crate::quant_fwht::quantize_mq4g256;
    use crate::quant_mq::quantize_mq2g256_lloyd;

    /// Deterministic Box-Muller-from-LCG Gaussian sampler — no external dep.
    /// Returns N samples with zero mean and unit variance.
    pub(crate) fn gaussian_samples(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut step = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as u64 & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut u1 = step() as f64;
            if u1 < 1e-12 {
                u1 = 1e-12;
            }
            let u2 = step() as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out.push((r * theta.cos()) as f32);
            if out.len() < n {
                out.push((r * theta.sin()) as f32);
            }
        }
        out
    }

    pub(crate) fn mse(a: &[f32], b: &[f32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = *x as f64 - *y as f64;
            acc += d * d;
        }
        acc / a.len() as f64
    }

    pub(crate) fn run_one_distribution(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        // Unit column weights — what DeepSeek V4's mq2-gptq-all build passes.
        let unit: Vec<f32> = vec![1.0; n];

        eprintln!("\n=== {label} (n={n}) ===");

        let lloyd_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let lloyd_recon = dequantize_mq2g256_lloyd_to_f32(&lloyd_bytes, n, &signs1, &signs2);
        let lloyd_mse = mse(weights, &lloyd_recon);
        eprintln!("  Lloyd                  MSE = {:.6e}", lloyd_mse);

        for damping in [0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let gptq_bytes =
                quantize_mq2g256_lloyd_gptq_with_damping(weights, &unit, &signs1, &signs2, damping);
            let gptq_recon = dequantize_mq2g256_lloyd_to_f32(&gptq_bytes, n, &signs1, &signs2);
            let gptq_mse = mse(weights, &gptq_recon);
            let delta = ((gptq_mse - lloyd_mse) / lloyd_mse) * 100.0;
            eprintln!(
                "  GPTQ d={damping:>4.1}             MSE = {:.6e}  ({:+.2}% vs Lloyd)",
                gptq_mse, delta
            );
        }
    }

    /// Does MQ2-Lloyd preserve weight magnitude, or does it shrink?
    ///
    /// This is the **routed-expert** tier (qt=19) in every DeepSeek V4 build,
    /// and it is exactly the branch `route_scale` multiplies — the shared
    /// expert and `ffn.gate` sit on a different tier and are untouched by it.
    ///
    /// Motivation: DS4 ships `route_scale` 1.8 (mq2r) and 2.2 (other builds)
    /// where the checkpoint declares 1.5, and the PyTorch reference scores
    /// PPL 4.693 at 1.5 — so 1.5 is correct for the model and our routed branch
    /// is systematically weak. Lloyd-Max centroids are conditional means, so by
    /// the orthogonality principle `E[w_hat.w] = E[w_hat^2]`, which makes the
    /// reconstruction provably SHORTER than the source: retained energy is
    /// `1 - MSE/E[w^2]` and the norm ratio is its square root. If that shortfall
    /// is large it is a candidate cause, and a global scalar can only ever
    /// approximate its average — a per-group gain would correct it properly.
    ///
    /// The sibling E8 codec was measured at retained ~0.999 (no shrinkage), so
    /// this tier is the one that matters. Informational: asserts sanity only
    /// and prints under `--nocapture`.
    #[test]
    pub(crate) fn mq2_lloyd_shrinkage_on_routed_expert_tier() {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        // Deterministic Gaussian; expert weights are near-Gaussian after the
        // FWHT incoherence rotation, which is the domain this codec is tuned to.
        let mut s: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let n = 256 * 512;
        let mut w = vec![0.0f32; n];
        for v in w.iter_mut() {
            let u1 = next().max(1e-12);
            let u2 = next();
            *v = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        }

        let recon = dequantize_mq2g256_lloyd_to_f32(
            &quantize_mq2g256_lloyd(&w, &signs1, &signs2),
            n,
            &signs1,
            &signs2,
        );

        let (mut dot, mut ss_w, mut ss_h) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            dot += f64::from(recon[i]) * f64::from(w[i]);
            ss_w += f64::from(w[i]) * f64::from(w[i]);
            ss_h += f64::from(recon[i]) * f64::from(recon[i]);
        }
        let retained = dot / ss_w;
        let norm_ratio = (ss_h / ss_w).sqrt();
        let energy_gain = 1.0 / retained;

        eprintln!("\nMQ2-Lloyd shrinkage on the routed-expert tier (n={n})");
        eprintln!("  retained  E[wh.w]/E[w^2] = {retained:.4}");
        eprintln!("  norm      |wh|/|w|       = {norm_ratio:.4}");
        eprintln!("  gain to restore energy   = {energy_gain:.4}");
        eprintln!("  shipped route_scale ratios: 1.8/1.5 = 1.2000, 2.2/1.5 = 1.4667\n");
        // Per-group spread decides whether a per-group gain beats a global one.
        // If every group shrinks identically, route_scale is already adequate
        // and a codebook change buys nothing; if the spread is wide, a global
        // scalar necessarily over-corrects some experts and under-corrects
        // others, and only a per-group gain fixes all of them.
        let mut per_group: Vec<f64> = Vec::with_capacity(n / 256);
        for g in 0..n / 256 {
            let (mut d, mut sw) = (0.0f64, 0.0f64);
            for i in g * 256..(g + 1) * 256 {
                d += f64::from(recon[i]) * f64::from(w[i]);
                sw += f64::from(w[i]) * f64::from(w[i]);
            }
            if sw > 0.0 {
                per_group.push(d / sw);
            }
        }
        per_group.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let gmean = per_group.iter().sum::<f64>() / per_group.len() as f64;
        let gsd = (per_group.iter().map(|v| (v - gmean).powi(2)).sum::<f64>()
            / per_group.len() as f64)
            .sqrt();
        let pick = |q: f64| per_group[((per_group.len() - 1) as f64 * q) as usize];
        eprintln!(
            "  per-group retained: mean {gmean:.4} sd {gsd:.4}  \
             min {:.4} p05 {:.4} p50 {:.4} p95 {:.4} max {:.4}",
            per_group[0],
            pick(0.05),
            pick(0.50),
            pick(0.95),
            per_group[per_group.len() - 1]
        );
        eprintln!(
            "  spread as % of mean: sd {:.2}%, p95-p05 {:.2}%\n",
            100.0 * gsd / gmean,
            100.0 * (pick(0.95) - pick(0.05)) / gmean
        );

        assert!(
            retained > 0.3 && retained < 1.3,
            "retained energy {retained} is not a sane round-trip"
        );
        assert!(norm_ratio.is_finite() && norm_ratio > 0.0);
    }

    /// Variant of plain Lloyd with tunable iteration count. Used to test
    /// whether the production 8-iter cap is leaving headroom.
    pub(crate) fn quantize_mq2g256_lloyd_niter(
        f32_data: &[f32],
        signs1: &[f32],
        signs2: &[f32],
        max_iter: usize,
    ) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        let mut sums = [0.0f64; 4];
                        let mut counts = [0u32; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev_assignments[i] != best as u8 {
                                changed += 1;
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                            sums[best] += w as f64;
                            counts[best] += 1;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if counts[k] > 0 {
                                cb[k] = (sums[k] / counts[k] as f64) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    pub(crate) fn run_lloyd_iter_sweep(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (n={n}) — Lloyd iteration sweep ===");
        let mut prev = f64::NAN;
        for niter in [1usize, 2, 4, 8, 16, 32, 64] {
            let bytes = quantize_mq2g256_lloyd_niter(weights, &signs1, &signs2, niter);
            let recon = dequantize_mq2g256_lloyd_to_f32(&bytes, n, &signs1, &signs2);
            let m = mse(weights, &recon);
            let delta = if prev.is_finite() {
                format!("  ({:+.3}% vs niter=prev)", ((m - prev) / prev) * 100.0)
            } else {
                String::new()
            };
            eprintln!("  Lloyd niter={niter:>3}        MSE = {m:.6e}{delta}");
            prev = m;
        }
    }

    /// Huber-Lloyd: same Lloyd loop but the centroid update is the
    /// weighted-mean of points with |w - cb| ≤ k_huber * sigma, where
    /// sigma is the within-cluster standard deviation. Points with
    /// larger residuals get clipped (treated as `cb ± k_huber * sigma`)
    /// so they don't drag centroids toward outlier values. With FWHT-
    /// rotated weights the long tails are dampened but not eliminated;
    /// this tests whether residual heavy-tailedness is hurting MSE.
    pub(crate) fn quantize_mq2g256_huber_lloyd(
        f32_data: &[f32],
        signs1: &[f32],
        signs2: &[f32],
        k_huber: f32,
        max_iter: usize,
    ) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        // Assignment pass — same as plain Lloyd.
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                        }
                        // Within-cluster sigma estimate (one pass).
                        let mut sums = [0.0f64; 4];
                        let mut sqs = [0.0f64; 4];
                        let mut cnts = [0u32; 4];
                        for i in 0..256 {
                            let k = indices[i] as usize;
                            let d = (group[i] - cb[k]) as f64;
                            sums[k] += group[i] as f64;
                            sqs[k] += d * d;
                            cnts[k] += 1;
                        }
                        let mut sigma = [0.0f64; 4];
                        for k in 0..4 {
                            if cnts[k] > 0 {
                                sigma[k] = (sqs[k] / cnts[k] as f64).sqrt();
                            }
                        }
                        // Huber-clipped update.
                        let mut wsums = [0.0f64; 4];
                        let mut wcnts = [0.0f64; 4];
                        for i in 0..256 {
                            let k = indices[i] as usize;
                            let lim = (k_huber as f64) * sigma[k].max(1e-9);
                            let resid = (group[i] - cb[k]) as f64;
                            let clipped = resid.max(-lim).min(lim);
                            let effective_w = cb[k] as f64 + clipped;
                            wsums[k] += effective_w;
                            wcnts[k] += 1.0;
                        }
                        let mut changed = 0u32;
                        for k in 0..4 {
                            if wcnts[k] > 0.0 {
                                let new_cb = (wsums[k] / wcnts[k]) as f32;
                                if new_cb != cb[k] {
                                    changed += 1;
                                }
                                cb[k] = new_cb;
                            }
                        }
                        // Suppress unused warnings on sums.
                        let _ = sums;
                        if it > 0 && changed == 0 {
                            break;
                        }
                    }
                    // Final argmin pass to lock indices to the final centroids.
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        indices[i] = best as u8;
                    }
                }
                // Sort centroids, remap, pack.
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    pub(crate) fn run_huber_sweep(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (n={n}) — Huber-Lloyd sweep (16 iter) ===");
        // Reference: plain Lloyd at 16 iter.
        let ref_bytes = quantize_mq2g256_lloyd_niter(weights, &signs1, &signs2, 16);
        let ref_recon = dequantize_mq2g256_lloyd_to_f32(&ref_bytes, n, &signs1, &signs2);
        let ref_mse = mse(weights, &ref_recon);
        eprintln!("  Lloyd (niter=16)          MSE = {ref_mse:.6e}");
        for k_huber in [1.0_f32, 1.5, 2.0, 2.5, 3.0, 10.0] {
            let bytes = quantize_mq2g256_huber_lloyd(weights, &signs1, &signs2, k_huber, 16);
            let recon = dequantize_mq2g256_lloyd_to_f32(&bytes, n, &signs1, &signs2);
            let m = mse(weights, &recon);
            let delta = ((m - ref_mse) / ref_mse) * 100.0;
            eprintln!(
                "  Huber k={k_huber:>4.1} (niter=16)   MSE = {m:.6e}  ({delta:+.2}% vs Lloyd16)"
            );
        }
    }

    /// GPTQ sequential pass on already-FWHT'd weights, no inner FWHT.
    /// Used to A/B test the FWHT-position hypothesis: production GPTQ
    /// FWHTs then propagates → noise injection. Pre-FWHT GPTQ
    /// (correlated input) should help when input weights have
    /// channel correlation.
    pub(crate) fn quantize_mq2g256_lloyd_gptq_no_fwht(
        f32_data: &[f32],
        damping: f32,
        max_iter: usize,
    ) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                // NO FWHT here — operate on raw correlated weights.
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                if range > 0.0 {
                    let mut prev = [0u8; 256];
                    for it in 0..max_iter {
                        let mut sums = [0.0f64; 4];
                        let mut counts = [0u32; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev[i] != best as u8 {
                                changed += 1;
                            }
                            prev[i] = best as u8;
                            sums[best] += w as f64;
                            counts[best] += 1;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if counts[k] > 0 {
                                cb[k] = (sums[k] / counts[k] as f64) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                }
                let cb_final = sorted_cb;
                // Sequential GPTQ with no inner FWHT.
                let mut indices = [0u8; 256];
                let mut residual = 0.0f32;
                for i in 0..256 {
                    let target = group[i] + residual;
                    let mut best = 0usize;
                    let mut best_d = (target - cb_final[0]).abs();
                    for k in 1..4 {
                        let d = (target - cb_final[k]).abs();
                        if d < best_d {
                            best_d = d;
                            best = k;
                        }
                    }
                    indices[i] = best as u8;
                    let err = target - cb_final[best];
                    residual = err * damping;
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(cb_final[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    /// Dequant the no-FWHT variant: indices + codebook, no inv-FWHT step.
    pub(crate) fn dequant_no_fwht(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let global_i = b * 256 + 4 * i + j;
                    if global_i < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[global_i] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    pub(crate) fn correlated_weights(n: usize, seed: u64, decay: f32) -> Vec<f32> {
        // AR(1) process: x_t = decay * x_{t-1} + sqrt(1 - decay^2) * z_t.
        // Produces channel-correlated weights (decay > 0).
        let gauss = gaussian_samples(n, seed);
        let mut out = Vec::with_capacity(n);
        let mut prev = 0.0f32;
        let noise_scale = (1.0f32 - decay * decay).sqrt();
        for &g in &gauss {
            let v = decay * prev + noise_scale * g;
            out.push(v);
            prev = v;
        }
        out
    }

    /// Dequant for MQ3-Lloyd (qt=20): 16 B fp16 codebook (8 entries) +
    /// 96 B 3-bit packed indices = 112 B / 256 weights.
    pub(crate) fn dequantize_mq3g256_lloyd_to_f32(
        data: &[u8],
        n_weights: usize,
        signs1: &[f32],
        signs2: &[f32],
    ) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 112;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        assert!(data.len() >= n_blocks * block_bytes);
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let mut cb = [0.0f32; 8];
            for k in 0..8 {
                cb[k] = f16_to_f32(u16::from_le_bytes([blk[2 * k], blk[2 * k + 1]]));
            }
            let mut group = [0.0f32; 256];
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = blk[bo];
                let b1 = blk[bo + 1];
                let b2 = blk[bo + 2];
                let mut q = [0u8; 8];
                q[0] = b0 & 7;
                q[1] = (b0 >> 3) & 7;
                q[2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2);
                q[3] = (b1 >> 1) & 7;
                q[4] = (b1 >> 4) & 7;
                q[5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1);
                q[6] = (b2 >> 2) & 7;
                q[7] = (b2 >> 5) & 7;
                for j in 0..8 {
                    group[chunk * 8 + j] = cb[q[j] as usize];
                }
            }
            cpu_inv_fwht_256(&mut group, signs1, signs2);
            let actual = (n_weights - b * 256).min(256);
            for j in 0..actual {
                out[b * 256 + j] = group[j];
            }
        }
        out
    }

    /// Quantifies the MSE cost of antirez's MQ3 → MQ2 down-projection
    /// downgrade. Procedure: take a synthetic DeepSeek V4-realistic weight
    /// distribution, quantize via MQ3-Lloyd (treat its dequant as the
    /// best-fit-available reference), then RE-quantize that dequant via
    /// MQ2-Lloyd. MSE delta = "what antirez loses by dropping MQ3 down".
    ///
    /// Result feeds the question: is the antirez precision tax (2/3 × MQ2
    /// + 1/3 × MQ3 ≈ 2.7 bpw vs 2.25 bpw all-MQ2, ~13 GB on a 256-expert
    /// 43-layer DeepSeek V4) buying meaningful per-tensor MSE reduction, or is
    /// the antirez win at high ctx mostly from Q8 attention?
    pub(crate) fn antirez_downgrade_cost(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        let mq3_bytes = quantize_mq3g256_lloyd(weights, &signs1, &signs2);
        let mq3_recon = dequantize_mq3g256_lloyd_to_f32(&mq3_bytes, n, &signs1, &signs2);
        let mq2_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let mq2_recon = dequantize_mq2g256_lloyd_to_f32(&mq2_bytes, n, &signs1, &signs2);
        // Direct MSE against the synthetic input (ground truth):
        let mq3_mse = mse(weights, &mq3_recon);
        let mq2_mse = mse(weights, &mq2_recon);
        let downgrade_pct = ((mq2_mse - mq3_mse) / mq3_mse) * 100.0;
        eprintln!("  {label} (n={n})");
        eprintln!("    MQ3-Lloyd (3.5 bpw) MSE = {mq3_mse:.6e}");
        eprintln!("    MQ2-Lloyd (2.25 bpw) MSE = {mq2_mse:.6e}");
        eprintln!("    MQ3→MQ2 downgrade cost: {downgrade_pct:+.1}% MSE");
    }

    #[test]
    pub(crate) fn antirez_mq3_to_mq2_downgrade_cost() {
        // Tests on the same DeepSeek V4-realistic distributions as the GPTQ probe.
        eprintln!("\n=== Antirez MQ3-down → MQ2-down downgrade cost ===");
        antirez_downgrade_cost("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        antirez_downgrade_cost("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        antirez_downgrade_cost("Sparse + outliers 16x256", &sw);
    }

    #[test]
    pub(crate) fn gptq_on_correlated_pre_fwht() {
        // The whole point of GPTQ is to exploit channel correlation.
        // Test it on correlated (decay=0.7), modestly-correlated (0.4),
        // and uncorrelated (0.0) inputs WITHOUT the inner FWHT step.
        //
        // If d>0 wins on correlated inputs but loses on uncorrelated,
        // that confirms: the production code's mistake is FWHT-then-GPTQ.
        // Fix path: drop the FWHT before the sequential pass (move it
        // into dequant or change the runtime kernel to apply it on
        // dequant'd values).
        eprintln!("\n=== GPTQ on correlated weights (no inner FWHT) ===");
        for (label, decay) in [
            ("decay=0.0 (uncorrelated)", 0.0f32),
            ("decay=0.4 (moderately correlated)", 0.4),
            ("decay=0.7 (strongly correlated)", 0.7),
            ("decay=0.9 (very correlated)", 0.9),
        ] {
            let n = 16 * 256;
            let w = correlated_weights(n, 0xc011a7ed, decay);
            // Reference: plain Lloyd via no-FWHT path with d=0.
            let ref_bytes = quantize_mq2g256_lloyd_gptq_no_fwht(&w, 0.0, 16);
            let ref_recon = dequant_no_fwht(&ref_bytes, n);
            let ref_mse = mse(&w, &ref_recon);
            eprintln!("\n  {label} (n={n})");
            eprintln!("    Lloyd                  MSE = {ref_mse:.6e}");
            for damping in [0.05f32, 0.1, 0.2, 0.3, 0.5, 0.8] {
                let b = quantize_mq2g256_lloyd_gptq_no_fwht(&w, damping, 16);
                let r = dequant_no_fwht(&b, n);
                let m = mse(&w, &r);
                let delta = ((m - ref_mse) / ref_mse) * 100.0;
                eprintln!(
                    "    GPTQ d={damping:>4.2} (no-fwht)   MSE = {m:.6e}  ({delta:+.2}% vs Lloyd)"
                );
            }
        }
    }

    #[test]
    pub(crate) fn huber_lloyd_headroom() {
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_huber_sweep("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_huber_sweep("Sparse + outliers 16x256", &sw);
        run_huber_sweep("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
    }

    /// Test "weight-norm proxy imatrix": a calibration-free approximation
    /// using column 2-norm of the weight matrix itself as the per-channel
    /// importance signal. Real AWQ uses sum_t |a_tj|^2; we substitute
    /// sum_i |w_ij|^2. Both produce a [K]-shaped vector that's used to
    /// weight the Lloyd codebook fit.
    ///
    /// If this gives meaningful MSE improvement over uniform Lloyd on
    /// heavy-tailed distributions, it's a viable calibration-free path
    /// to better DeepSeek V4 quants. Bench-falsified if it doesn't beat uniform
    /// by a clear margin.
    pub(crate) fn weight_norm_proxy_imatrix(weights: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut col_norms = vec![0.0f32; k];
        for r in 0..m {
            for j in 0..k {
                let w = weights[r * k + j];
                col_norms[j] += w * w;
            }
        }
        for v in col_norms.iter_mut() {
            *v = v.sqrt();
        }
        // Normalize so geometric mean is 1.0 (matches AWQ convention).
        let mut sum_log = 0.0f64;
        for &v in &col_norms {
            sum_log += (v.max(1e-12) as f64).ln();
        }
        let mean_log = sum_log / k as f64;
        for v in col_norms.iter_mut() {
            *v = ((*v as f64).ln() - mean_log).exp() as f32;
        }
        col_norms
    }

    pub(crate) fn run_weight_norm_proxy_sweep(label: &str, weights: &[f32], m: usize, k: usize) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (m={m}, k={k}, n={n}) ===");
        // Uniform Lloyd baseline.
        let ref_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let ref_recon = dequantize_mq2g256_lloyd_to_f32(&ref_bytes, n, &signs1, &signs2);
        let ref_mse = mse(weights, &ref_recon);
        eprintln!("  Uniform Lloyd                MSE = {ref_mse:.6e}");
        // Weight-norm proxy imatrix.
        let col_imatrix = weight_norm_proxy_imatrix(weights, m, k);
        let proxy_bytes = quantize_mq2g256_lloyd_weighted(weights, &col_imatrix, &signs1, &signs2);
        let proxy_recon = dequantize_mq2g256_lloyd_to_f32(&proxy_bytes, n, &signs1, &signs2);
        let proxy_mse = mse(weights, &proxy_recon);
        let delta = ((proxy_mse - ref_mse) / ref_mse) * 100.0;
        eprintln!(
            "  Weight-norm-proxy Lloyd      MSE = {proxy_mse:.6e}  ({delta:+.2}% vs uniform)"
        );
    }

    /// Quantize via Lloyd WITHOUT the FWHT step — Lloyd applied directly
    /// to the natural (pre-rotation) weight distribution. Same 4-codepoint
    /// codebook + 2-bit indices.
    pub(crate) fn quantize_mq2g256_lloyd_no_fwht(f32_data: &[f32]) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                // NO FWHT — Lloyd directly on natural distribution.
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let max_iter = 16;
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        let mut sums = [0.0f64; 4];
                        let mut counts = [0u32; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev_assignments[i] != best as u8 {
                                changed += 1;
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                            sums[best] += w as f64;
                            counts[best] += 1;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if counts[k] > 0 {
                                cb[k] = (sums[k] / counts[k] as f64) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    pub(crate) fn dequant_mq2_no_fwht(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let global_i = b * 256 + 4 * i + j;
                    if global_i < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[global_i] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    /// Quantize W (natural basis) with imatrix-weighted Lloyd, no FWHT.
    /// Returns (codebook, indices) — both in natural basis.
    pub(crate) fn lloyd_imatrix_no_fwht(weights: &[f32], col_weights: &[f32]) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = weights.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        let blocks_per_row = col_weights.len() / group_size;
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&weights[start..end]);
                // Use natural distribution; NO FWHT.
                let col_off = (b % blocks_per_row) * group_size;
                let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let max_iter = 16;
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        let mut wsums = [0.0f64; 4];
                        let mut wtotals = [0.0f64; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev_assignments[i] != best as u8 {
                                changed += 1;
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                            let pw = block_w[i] as f64;
                            wsums[best] += pw * w as f64;
                            wtotals[best] += pw;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if wtotals[k] > 0.0 {
                                cb[k] = (wsums[k] / wtotals[k]) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    pub(crate) fn dequant_no_fwht_natural(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let gi = b * 256 + 4 * i + j;
                    if gi < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[gi] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    pub(crate) fn gemv_f32(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for r in 0..m {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += w[r * k + j] as f64 * x[j] as f64;
            }
            y[r] = acc as f32;
        }
        y
    }

    #[test]
    pub(crate) fn prefwht_imatrix_lloyd_value() {
        // Activation-weighted A/B test of post-FWHT vs pre-FWHT imatrix-Lloyd.
        // Generate W [m=256, k=4096] with HETEROGENEOUS column variances —
        // some columns have stddev=3, others stddev=0.1. Imatrix captures the
        // ground-truth importance. Run a gemv with this W against a random
        // unit-Gaussian X, then compare gemv-error for the two quant methods.
        //
        // If pre-FWHT-imatrix-Lloyd reduces gemv error meaningfully on
        // activations vs post-FWHT, that's the green light for the
        // pre-FWHT-Lloyd refactor (Action 5 in playbook).
        let m = 256;
        let k = 4096;
        let n = m * k;

        // Build heterogeneous-column W: column j has scale = log-uniform in
        // [0.1, 3.0] — gives 30x spread, mimics real LLM channel importance.
        let mut w = gaussian_samples(n, 0xc011c011);
        let mut col_scales = vec![0.0f32; k];
        let mut state: u64 = 0xc0ffeeed;
        for j in 0..k {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64;
            // log-uniform in [0.1, 3.0]
            col_scales[j] = (0.1_f64.ln() + u * (3.0_f64.ln() - 0.1_f64.ln())).exp() as f32;
        }
        for r in 0..m {
            for j in 0..k {
                w[r * k + j] *= col_scales[j];
            }
        }
        // Imatrix: per-column 2-norm of W (mimics what a real activation
        // imatrix produces — bigger for important channels). Geomean-normalize.
        let mut imatrix = vec![0.0f32; k];
        for j in 0..k {
            let mut sum2 = 0.0f64;
            for r in 0..m {
                sum2 += (w[r * k + j] as f64).powi(2);
            }
            imatrix[j] = sum2.sqrt() as f32;
        }
        let mut sum_log = 0.0f64;
        for &v in &imatrix {
            sum_log += (v.max(1e-12) as f64).ln();
        }
        let mean_log = sum_log / k as f64;
        for v in imatrix.iter_mut() {
            *v = ((*v as f64).ln() - mean_log).exp() as f32;
        }

        // Random unit-Gaussian X for activations.
        let x = gaussian_samples(k, 0xacd1ac);
        let y_ref = gemv_f32(&w, &x, m, k);

        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        // METHOD A: post-FWHT imatrix-Lloyd (production).
        let bytes_a = quantize_mq2g256_lloyd_weighted(&w, &imatrix, &signs1, &signs2);
        let recon_a = dequantize_mq2g256_lloyd_to_f32(&bytes_a, n, &signs1, &signs2);
        let y_a = gemv_f32(&recon_a, &x, m, k);
        let err_a: f64 = y_ref
            .iter()
            .zip(y_a.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        // METHOD B: pre-FWHT imatrix-Lloyd (proposed refactor).
        let bytes_b = lloyd_imatrix_no_fwht(&w, &imatrix);
        let recon_b = dequant_no_fwht_natural(&bytes_b, n);
        let y_b = gemv_f32(&recon_b, &x, m, k);
        let err_b: f64 = y_ref
            .iter()
            .zip(y_b.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        // METHOD C: post-FWHT uniform Lloyd (current production w/o imatrix).
        let bytes_c = quantize_mq2g256_lloyd(&w, &signs1, &signs2);
        let recon_c = dequantize_mq2g256_lloyd_to_f32(&bytes_c, n, &signs1, &signs2);
        let y_c = gemv_f32(&recon_c, &x, m, k);
        let err_c: f64 = y_ref
            .iter()
            .zip(y_c.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        eprintln!("\n=== Pre-FWHT vs post-FWHT imatrix-Lloyd (activation-weighted) ===");
        eprintln!("  W shape [{m}, {k}], heterogeneous column variances (0.1-3.0x)");
        eprintln!("  Method A: post-FWHT imatrix-Lloyd (current prod)   gemv MSE = {err_a:.6e}");
        eprintln!("  Method B: pre-FWHT  imatrix-Lloyd (proposed)       gemv MSE = {err_b:.6e}");
        eprintln!("  Method C: post-FWHT uniform Lloyd (no imatrix)     gemv MSE = {err_c:.6e}");
        eprintln!();
        let ab = ((err_b - err_a) / err_a) * 100.0;
        let ac = ((err_a - err_c) / err_c) * 100.0;
        let bc = ((err_b - err_c) / err_c) * 100.0;
        eprintln!("  Δ A→B (pre-FWHT win):              {ab:+.2}%");
        eprintln!("  Δ C→A (current imatrix vs uniform):{ac:+.2}%");
        eprintln!("  Δ C→B (pre-FWHT vs uniform):       {bc:+.2}%");
    }

    #[test]
    pub(crate) fn fwht_value_audit() {
        // Hypothesis: FWHT-rotation makes Lloyd more accurate because the
        // rotation decorrelates weights toward a Gaussian distribution, and
        // Lloyd's 4 codepoints are MSE-optimal for Gaussian.
        //
        // Test: quantize the SAME synthetic distribution two ways:
        //   A) Lloyd with FWHT (production path)
        //   B) Lloyd without FWHT (natural distribution)
        // Compute MSE for each. If FWHT wins consistently, the rotation is
        // earning its complexity. If they're close, dropping FWHT unblocks
        // proper imatrix integration (per
        // project_lloyd_imatrix_fwht_channel_mixing).
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        let cases: &[(&str, Box<dyn Fn() -> Vec<f32>>)] = &[
            (
                "Gaussian 16x256",
                Box::new(|| gaussian_samples(16 * 256, 0xc001cafe)),
            ),
            (
                "Heavy-tailed 16x256",
                Box::new(|| {
                    let mut htw = gaussian_samples(16 * 256, 0xfeed);
                    let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
                    for (i, t) in tail.iter().enumerate() {
                        htw[i * 20] = t * 3.0;
                    }
                    htw
                }),
            ),
            (
                "Sparse + outliers 16x256",
                Box::new(|| {
                    let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
                    for v in sw.iter_mut() {
                        *v *= 0.1;
                    }
                    for i in 0..(16 * 256 / 20) {
                        sw[i * 20] *= 30.0;
                    }
                    sw
                }),
            ),
            (
                "Bimodal (50% near -1, 50% near +1)",
                Box::new(|| {
                    let mut bw = gaussian_samples(16 * 256, 0xb1ba1);
                    for (i, v) in bw.iter_mut().enumerate() {
                        *v = 0.3 * *v + if i % 2 == 0 { -1.0 } else { 1.0 };
                    }
                    bw
                }),
            ),
        ];

        eprintln!("\n=== FWHT value audit ===");
        eprintln!(
            "{:35} {:>14} {:>14} {:>10}",
            "distribution", "fwht MSE", "no-fwht MSE", "fwht win %"
        );
        for (label, generate) in cases {
            let w = generate();
            let n = w.len();
            let fwht_bytes = quantize_mq2g256_lloyd(&w, &signs1, &signs2);
            let fwht_recon = dequantize_mq2g256_lloyd_to_f32(&fwht_bytes, n, &signs1, &signs2);
            let fwht_mse = mse(&w, &fwht_recon);
            let nofwht_bytes = quantize_mq2g256_lloyd_no_fwht(&w);
            let nofwht_recon = dequant_mq2_no_fwht(&nofwht_bytes, n);
            let nofwht_mse = mse(&w, &nofwht_recon);
            let win_pct = ((nofwht_mse - fwht_mse) / nofwht_mse) * 100.0;
            eprintln!(
                "{:35} {:14.6e} {:14.6e} {:+9.2}%",
                label, fwht_mse, nofwht_mse, win_pct
            );
        }
    }

    #[test]
    pub(crate) fn weight_norm_proxy_imatrix_sweep() {
        // Generate synthetic [m, k] matrices that mimic DeepSeek V4's expert
        // shapes (m=2048, k=4096 for gate; m=4096, k=2048 for down).
        // Use heavy-tailed and sparse-outlier variants to stress the
        // proxy.
        let m = 2048;
        let k = 4096;
        let n = m * k;
        eprintln!("\n=== Weight-norm proxy imatrix sweep ===");
        run_weight_norm_proxy_sweep(
            "Gaussian [2048, 4096]",
            &gaussian_samples(n, 0xc001cafe),
            m,
            k,
        );
        // Heavy-tailed: 5% of weights drawn from N(0, 3).
        let mut htw = gaussian_samples(n, 0xfeed);
        let tail_count = n / 20;
        let tail = gaussian_samples(tail_count, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_weight_norm_proxy_sweep("Heavy-tailed [2048, 4096]", &htw, m, k);
        // Per-column variance heterogeneity: make column j scale with j/k.
        let mut col_het = gaussian_samples(n, 0xc011c011);
        for r in 0..m {
            for j in 0..k {
                let scale = 0.1 + 1.9 * (j as f32 / k as f32);
                col_het[r * k + j] *= scale;
            }
        }
        run_weight_norm_proxy_sweep("Per-column var heterogeneity", &col_het, m, k);
    }

    #[test]
    pub(crate) fn lloyd_iteration_headroom() {
        // The production 8-iter cap may or may not converge on heavy-tailed
        // distributions. Sweep niter ∈ {1, 2, 4, 8, 16, 32, 64} to find the
        // convergence floor — if 32 or 64 iter gives meaningfully lower
        // MSE than 8, that's free headroom (offline quant cost only).
        run_lloyd_iter_sweep("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_lloyd_iter_sweep("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_lloyd_iter_sweep("Sparse + outliers 16x256", &sw);
    }

    #[test]
    pub(crate) fn sweep_deepseek4_like_distributions() {
        // 1) Pure Gaussian — baseline.
        run_one_distribution("N(0,1), 256 weights", &gaussian_samples(256, 0xc001cafe));

        // 2) Pure Gaussian, larger sample — averages across multiple blocks.
        run_one_distribution(
            "N(0,1), 16x256 weights",
            &gaussian_samples(16 * 256, 0xc001cafe),
        );

        // 3) Heavy-tailed mixture — 5% from N(0, 3), rest N(0, 1).
        //    Mimics DeepSeek V4's expert distributions with occasional outliers.
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            // Sprinkle the tail in every 20th slot.
            htw[i * 20] = t * 3.0;
        }
        run_one_distribution("Heavy-tailed, 16x256 weights", &htw);

        // 4) Sparse weights — most near zero, a few large. Sometimes
        //    happens in attention-related projections.
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        // Inject 5% large values.
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_one_distribution("Sparse (10% scale, 5% × 30 outliers)", &sw);
    }
}

/// Real-DeepSeek V4 per-block diagnostic. Reads an HFQ file directly via memmap2
/// (bypasses the hipfire-runtime hfq reader which currently has a broken
/// arch dep — keeps this probe self-contained inside hipfire-quantize).
/// For each MQ2-Lloyd (qt=19) and MQ3-Lloyd (qt=20) tensor, samples up to
/// MAX_SAMPLE_BLOCKS blocks and computes per-block stats:
///   - codebook range (max_cb - min_cb)
///   - codepoint spacing variance (how uneven the codebook is)
///   - index entropy (uniform = 2 bits for MQ2, log2(8)=3 for MQ3)
/// Then ranks tensors by mean per-block range to identify which tensors
/// have the highest dynamic range (= hardest to compress at given bpw).
///
/// Run with: cargo test --release -p hipfire-quantize --
///           --ignored hfq_block_range_diag -- --nocapture
///
/// Reads path from HIPFIRE_QUANT_DIAG_PATH env var (default points at
/// a local DeepSeek V4 HFQ snapshot).
#[cfg(test)]
mod hfq_block_diag {
    use super::*;
    use memmap2::Mmap;
    use std::fs::File;
    use std::path::Path;

    pub(crate) struct TensorInfo {
        name: String,
        quant_type: u8,
        shape: Vec<u32>,
        data_offset: usize,
        data_size: usize,
    }

    pub(crate) fn parse_hfq_metadata(path: &Path) -> std::io::Result<String> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        assert_eq!(&mmap[0..4], b"HFQM");
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut json_end = 0usize;
        for (i, &b) in mmap[metadata_offset..data_offset].iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                if b == b'\\' {
                    esc = true;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            if b == b'"' {
                in_str = true;
                continue;
            }
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
        Ok(String::from_utf8_lossy(&mmap[metadata_offset..metadata_offset + json_end]).to_string())
    }

    #[test]
    #[ignore]
    pub(crate) fn hfq_dump_metadata() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        let json = parse_hfq_metadata(path).expect("parse");
        // Print just keys at top level + any "source" / "path" / "input" hints.
        eprintln!("=== Metadata from {path:?} (top 2000 chars) ===");
        let truncated: String = json.chars().take(2000).collect();
        eprintln!("{}", truncated);
        if json.len() > 2000 {
            eprintln!("... ({} chars total)", json.len());
        }
    }

    pub(crate) fn parse_hfq(path: &Path) -> std::io::Result<(Mmap, Vec<TensorInfo>)> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        assert_eq!(&mmap[0..4], b"HFQM", "Not HFQ");
        let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        // Find JSON end by brace-matching.
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut json_end = 0usize;
        for (i, &b) in mmap[metadata_offset..data_offset].iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                if b == b'\\' {
                    esc = true;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            if b == b'"' {
                in_str = true;
                continue;
            }
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
        let mut pos = metadata_offset + json_end;
        let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        assert_eq!(idx_n, n_tensors);
        pos += 4;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut cum = data_offset;
        for _ in 0..n_tensors {
            let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).into_owned();
            pos += name_len;
            let quant_type = mmap[pos];
            pos += 1;
            let n_dims = mmap[pos] as usize;
            pos += 1;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            // Skip group_size u32.
            pos += 4;
            let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            tensors.push(TensorInfo {
                name,
                quant_type,
                shape,
                data_offset: cum,
                data_size,
            });
            cum += data_size;
        }
        Ok((mmap, tensors))
    }

    pub(crate) fn classify(name: &str) -> &'static str {
        if name.contains("ffn.experts.") && name.ends_with("w1.weight") {
            return "routed.w1 (gate)";
        }
        if name.contains("ffn.experts.") && name.ends_with("w2.weight") {
            return "routed.w2 (down)";
        }
        if name.contains("ffn.experts.") && name.ends_with("w3.weight") {
            return "routed.w3 (up)";
        }
        if name.contains("shared_experts.w1") {
            return "shared.w1";
        }
        if name.contains("shared_experts.w2") {
            return "shared.w2";
        }
        if name.contains("shared_experts.w3") {
            return "shared.w3";
        }
        if name.ends_with("attn.wq_a.weight") || name.ends_with("attn.wq_b.weight") {
            return "attn.q";
        }
        if name.ends_with("attn.wkv.weight") {
            return "attn.kv";
        }
        if name.ends_with("attn.wo_a.weight") || name.ends_with("attn.wo_b.weight") {
            return "attn.wo";
        }
        if name.contains("compressor.wkv") || name.contains("compressor.wgate") {
            return "compressor";
        }
        if name.contains("indexer.") {
            return "indexer";
        }
        "other"
    }

    /// Stats per block at MQ2 (4 codepoints, 8 B codebook + 64 B indices = 72 B/group).
    pub(crate) fn block_stats_mq2(data: &[u8]) -> Option<(f32, f32, f32)> {
        if data.len() < 8 {
            return None;
        }
        let mut cb = [0.0f32; 4];
        for k in 0..4 {
            cb[k] = f16_to_f32(u16::from_le_bytes([data[2 * k], data[2 * k + 1]]));
        }
        let lo = cb.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let mean = cb.iter().sum::<f32>() / 4.0;
        let spacing_var = cb.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / 4.0;
        // Index histogram.
        let mut hist = [0u32; 4];
        for i in 0..64 {
            let b = data[8 + i];
            for j in 0..4 {
                hist[((b >> (j * 2)) & 0x3) as usize] += 1;
            }
        }
        let total: u32 = hist.iter().sum();
        let mut h_bits = 0.0f32;
        for &c in &hist {
            if c > 0 {
                let p = c as f32 / total as f32;
                h_bits -= p * p.log2();
            }
        }
        Some((range, spacing_var, h_bits))
    }

    /// Stats per block at MQ3 (8 codepoints, 16 B codebook + 96 B indices = 112 B/group).
    pub(crate) fn block_stats_mq3(data: &[u8]) -> Option<(f32, f32, f32)> {
        if data.len() < 16 {
            return None;
        }
        let mut cb = [0.0f32; 8];
        for k in 0..8 {
            cb[k] = f16_to_f32(u16::from_le_bytes([data[2 * k], data[2 * k + 1]]));
        }
        let lo = cb.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let mean = cb.iter().sum::<f32>() / 8.0;
        let spacing_var = cb.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / 8.0;
        // Reconstruct indices.
        let mut hist = [0u32; 8];
        for chunk in 0..32 {
            let bo = 16 + chunk * 3;
            let b0 = data[bo];
            let b1 = data[bo + 1];
            let b2 = data[bo + 2];
            let q = [
                b0 & 7,
                (b0 >> 3) & 7,
                ((b0 >> 6) & 3) | ((b1 & 1) << 2),
                (b1 >> 1) & 7,
                (b1 >> 4) & 7,
                ((b1 >> 7) & 1) | ((b2 & 3) << 1),
                (b2 >> 2) & 7,
                (b2 >> 5) & 7,
            ];
            for v in q {
                hist[v as usize] += 1;
            }
        }
        let total: u32 = hist.iter().sum();
        let mut h_bits = 0.0f32;
        for &c in &hist {
            if c > 0 {
                let p = c as f32 / total as f32;
                h_bits -= p * p.log2();
            }
        }
        Some((range, spacing_var, h_bits))
    }

    pub(crate) fn cpu_inv_fwht_local(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
        super::cpu_inv_fwht_256(x, signs1, signs2);
    }

    pub(crate) fn dequant_mq3_lloyd(
        data: &[u8],
        n_weights: usize,
        signs1: &[f32],
        signs2: &[f32],
    ) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 112;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let mut cb = [0.0f32; 8];
            for k in 0..8 {
                cb[k] = f16_to_f32(u16::from_le_bytes([blk[2 * k], blk[2 * k + 1]]));
            }
            let mut group = [0.0f32; 256];
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = blk[bo];
                let b1 = blk[bo + 1];
                let b2 = blk[bo + 2];
                let q = [
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) & 3) | ((b1 & 1) << 2),
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) & 1) | ((b2 & 3) << 1),
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ];
                for j in 0..8 {
                    group[chunk * 8 + j] = cb[q[j] as usize];
                }
            }
            cpu_inv_fwht_local(&mut group, signs1, signs2);
            let actual = (n_weights - b * 256).min(256);
            for j in 0..actual {
                out[b * 256 + j] = group[j];
            }
        }
        out
    }

    pub(crate) fn qt_name(qt: u8) -> &'static str {
        match qt {
            1 => "F16",
            2 => "F32",
            3 => "Q8F16",
            5 => "Q8HFQ",
            6 => "HFQ4G256",
            7 => "HFQ4G128",
            13 => "MQ4G256",
            14 => "MQ8G256",
            15 => "MQ6G256",
            17 => "MQ3G256",
            18 => "MQ2G256",
            19 => "MQ2G256Lloyd",
            20 => "MQ3G256Lloyd",
            21 => "HFP4G32",
            24 => "MFP4G32",
            34 => "MFP4G32E8",
            35 => "MFP4G32E8SOA",
            36 => "MFP3G32E8",
            37 => "MFP2G32E8",
            _ => "?",
        }
    }

    /// Sample a real DeepSeek V4 MQ2-Lloyd tensor, dequant a few blocks, and
    /// report the distribution shape. Compares against the synthetic
    /// distributions used in fwht_value_audit + GPTQ probes to see which
    /// our DeepSeek V4 weights actually resemble.
    #[test]
    #[ignore]
    pub(crate) fn hfq_dist_sample() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        let (mmap, tensors) = parse_hfq(path).expect("parse hfq");

        // Take 8 different routed-expert tensors (w1, w2, w3 from a few
        // layers/experts) and one attention tensor + one shared tensor.
        let sample_names = [
            "layers.5.ffn.experts.0.w1.weight",   // gate (mid layer)
            "layers.5.ffn.experts.0.w2.weight",   // down
            "layers.5.ffn.experts.0.w3.weight",   // up
            "layers.20.ffn.experts.50.w1.weight", // gate (later layer)
            "layers.20.ffn.experts.50.w2.weight",
            "layers.40.ffn.experts.100.w2.weight", // down (deep layer)
            "layers.5.ffn.shared_experts.w2.weight", // shared down
            "layers.5.attn.wo_b.weight",           // attention output
        ];
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        eprintln!("\n=== Real DeepSeek V4 weight distribution stats (4096 weights per tensor) ===");
        eprintln!(
            "{:55} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "tensor", "qt", "mean", "stddev", "p99/sd", "kurtosis"
        );
        for sname in sample_names {
            let t_idx = tensors.iter().position(|t| t.name == sname);
            let t = match t_idx {
                Some(i) => &tensors[i],
                None => continue,
            };
            // Sample first 16 blocks = 4096 weights. Skip unsupported qts.
            let block_bytes = match t.quant_type {
                19 => 72,
                20 => 112,
                3 => 34,
                _ => {
                    eprintln!("  {:55} {:>2} (skip qt)", sname, t.quant_type);
                    continue;
                }
            };
            let n_blocks = (t.data_size / block_bytes).min(16);
            if n_blocks == 0 {
                continue;
            }
            let n_w = n_blocks * 256;
            let recon: Vec<f32> = if t.quant_type == 19 {
                let raw = &mmap[t.data_offset..t.data_offset + n_blocks * 72];
                super::dequantize_mq2g256_lloyd_to_f32(raw, n_w, &signs1, &signs2)
            } else if t.quant_type == 20 {
                let raw = &mmap[t.data_offset..t.data_offset + n_blocks * 112];
                dequant_mq3_lloyd(raw, n_w, &signs1, &signs2)
            } else {
                eprintln!(
                    "  {:55} {:>2} (unsupported qt for dequant, skipping)",
                    sname, t.quant_type
                );
                continue;
            };
            // Compute stats.
            let n = recon.len() as f64;
            let mean = recon.iter().map(|&x| x as f64).sum::<f64>() / n;
            let var = recon
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let stddev = var.sqrt();
            // Kurtosis (Pearson) — measures heavy-tailedness; Gaussian = 3.
            let mu4 = recon
                .iter()
                .map(|&x| (x as f64 - mean).powi(4))
                .sum::<f64>()
                / n;
            let kurt = mu4 / var.powi(2);
            // p99/sd — ratio of 99th percentile abs value to sd.
            let mut absvals: Vec<f64> = recon.iter().map(|&x| (x as f64 - mean).abs()).collect();
            absvals
                .sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99 = absvals[(absvals.len() * 99 / 100).min(absvals.len() - 1)];
            let p99_over_sd = p99 / stddev;
            eprintln!(
                "{:55} {:>2} {:>10.4e} {:>10.4e} {:>10.3} {:>10.3}",
                sname, t.quant_type, mean, stddev, p99_over_sd, kurt
            );
        }
        // Reference values from synthetic distributions:
        eprintln!("\nReference (synthetic):");
        eprintln!("  Gaussian:            p99/sd ≈ 2.33    kurtosis ≈ 3.0");
        eprintln!("  Heavy-tailed (5% × 3): p99/sd ≈ 2.5-3   kurtosis ≈ 3-6");
        eprintln!("  Sparse outliers:     p99/sd ≈ 10+     kurtosis ≈ 30+");
        eprintln!("  Bimodal:             p99/sd ≈ 1.5-2   kurtosis < 3 (platykurtic)");
    }

    #[test]
    #[ignore]
    pub(crate) fn hfq_inventory() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        eprintln!("opening {path:?}");
        let (_mmap, tensors) = parse_hfq(path).expect("parse hfq");
        eprintln!("{} tensors", tensors.len());
        // Bucket by (family, qt).
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<(&'static str, u8), (u64, u64)> = BTreeMap::new();
        let mut total_bytes: u64 = 0;
        for t in &tensors {
            let fam = classify(&t.name);
            let e = counts.entry((fam, t.quant_type)).or_default();
            e.0 += 1;
            e.1 += t.data_size as u64;
            total_bytes += t.data_size as u64;
        }
        eprintln!(
            "{:30} {:>14} {:>8} {:>14}",
            "family", "qt", "count", "bytes"
        );
        for ((fam, qt), (cnt, bytes)) in &counts {
            eprintln!(
                "{:30} {:>2} {:12} {:>8} {:>14}",
                fam,
                qt,
                qt_name(*qt),
                cnt,
                bytes
            );
        }
        eprintln!(
            "\ntotal data bytes: {} ({:.2} GiB)",
            total_bytes,
            total_bytes as f64 / (1024.0_f64.powi(3))
        );
    }

    #[test]
    #[ignore]
    pub(crate) fn hfq_block_range_diag() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash.mq2lloyd".to_string());
        let path = Path::new(&path_str);
        eprintln!("opening {path:?}");
        let (mmap, tensors) = parse_hfq(path).expect("parse hfq");
        eprintln!("{} tensors, file mapped", tensors.len());

        // Bucket by (family, qt) → list of (mean_range, mean_var, mean_entropy, n_blocks).
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<(&'static str, u8), Vec<(f32, f32, f32, usize)>> =
            BTreeMap::new();

        // Sample at most this many blocks per tensor; routed-expert tensors are
        // huge (~1 MB each in the layer's batched blob form, 256 experts × 43
        // layers = ~30k tensors). Cap CPU time.
        pub(crate) const MAX_BLOCKS_PER_TENSOR: usize = 64;

        for t in &tensors {
            if t.quant_type != 19 && t.quant_type != 20 {
                continue;
            }
            let block_bytes = if t.quant_type == 19 { 72 } else { 112 };
            let raw = &mmap[t.data_offset..t.data_offset + t.data_size];
            let n_blocks = t.data_size / block_bytes;
            if n_blocks == 0 {
                continue;
            }
            let stride = (n_blocks / MAX_BLOCKS_PER_TENSOR.min(n_blocks)).max(1);
            let mut sum_range = 0.0f64;
            let mut sum_var = 0.0f64;
            let mut sum_h = 0.0f64;
            let mut n_sampled = 0usize;
            let mut bi = 0;
            while bi < n_blocks {
                let blk = &raw[bi * block_bytes..(bi + 1) * block_bytes];
                let stats = if t.quant_type == 19 {
                    block_stats_mq2(blk)
                } else {
                    block_stats_mq3(blk)
                };
                if let Some((r, v, h)) = stats {
                    sum_range += r as f64;
                    sum_var += v as f64;
                    sum_h += h as f64;
                    n_sampled += 1;
                }
                bi += stride;
            }
            if n_sampled == 0 {
                continue;
            }
            let fam = classify(&t.name);
            buckets.entry((fam, t.quant_type)).or_default().push((
                (sum_range / n_sampled as f64) as f32,
                (sum_var / n_sampled as f64) as f32,
                (sum_h / n_sampled as f64) as f32,
                n_sampled,
            ));
        }

        eprintln!("\n=== Per-family block stats (sampled {MAX_BLOCKS_PER_TENSOR}/tensor) ===");
        eprintln!(
            "{:30} {:3} {:>6} {:>10} {:>10} {:>10}",
            "family", "qt", "tensors", "mean_range", "mean_var", "mean_entropy"
        );
        for ((fam, qt), entries) in &buckets {
            let n_tensors = entries.len();
            let mean_range =
                entries.iter().map(|(r, _, _, _)| *r as f64).sum::<f64>() / n_tensors as f64;
            let mean_var =
                entries.iter().map(|(_, v, _, _)| *v as f64).sum::<f64>() / n_tensors as f64;
            let mean_h =
                entries.iter().map(|(_, _, h, _)| *h as f64).sum::<f64>() / n_tensors as f64;
            eprintln!(
                "{:30} {:3} {:>6} {:>10.4} {:>10.4} {:>10.4}",
                fam, qt, n_tensors, mean_range, mean_var, mean_h
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_filter::{is_q8_tensor, q8_class_of, should_quantize};
    use crate::quant_fwht::{cpu_fwht_256, gen_fwht_signs};
    use crate::hfq::{kmap_resolve, kmap_resolve_mode, QuantLevel};

    /// The MQ*-G256-GL codebooks are NOT stored in the `.hfq` file: the encoder
    /// bakes them in via `gl_encode_block(&GL_CB2 | &GL_CB3, ..)` and the runtime
    /// re-supplies them as scalar kernel args from
    /// `rdna_compute::{GL_CB2, GL_CB3}`. Drift between the two arrays is a
    /// SILENT accuracy failure — every weight decodes to a plausible-but-wrong
    /// level and nothing errors. This test is the only thing standing between
    /// the two copies, so keep it.
    ///
    /// If it fails: fix whichever side you did NOT intend to change. Bumping the
    /// codebook is a FORMAT CHANGE — existing qt-38/39 `.hfq` files decode wrong
    /// against a new codebook, so it needs a new quant_type, not an edit.
    #[test]
    pub(crate) fn gl_codebooks_match_runtime() {
        assert_eq!(
            GL_CB2,
            rdna_compute::GL_CB2,
            "hipfire-quantize GL_CB2 has drifted from rdna_compute::GL_CB2 — \
             qt=38 weights would decode against the wrong 2-bit levels"
        );
        assert_eq!(
            GL_CB3,
            rdna_compute::GL_CB3,
            "hipfire-quantize GL_CB3 has drifted from rdna_compute::GL_CB3 — \
             qt=39 weights would decode against the wrong 3-bit levels"
        );
    }

    /// Pins the GL on-disk geometry the runtime loader + kernels assume:
    /// SoA `[m*gpr*IDX B indices][m*gpr*2 B fp16 scales]`, IDX = 64 (MQ2-GL) /
    /// 96 (MQ3-GL). The kernels derive the scale-region base as `M*gpr*IDX`, so
    /// a size change here is a silent read past/short of the scales.
    #[test]
    pub(crate) fn gl_blob_layout_matches_runtime_constants() {
        let (m, k) = (4usize, 512usize);
        let gpr = k / 256;
        let signs1 = vec![1.0f32; 256];
        let signs2 = vec![1.0f32; 256];
        let w: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();

        let b2 = quantize_mq2g256gl(&w, m, k, &signs1, &signs2);
        assert_eq!(
            b2.len(),
            m * gpr * (rdna_compute::GL_MQ2_GROUP_IDX_BYTES + rdna_compute::GL_GROUP_SCALE_BYTES),
            "MQ2G256GL blob size disagrees with GL_MQ2_GROUP_IDX_BYTES/GL_GROUP_SCALE_BYTES"
        );

        let b3 = quantize_mq3g256gl(&w, m, k, &signs1, &signs2);
        assert_eq!(
            b3.len(),
            m * gpr * (rdna_compute::GL_MQ3_GROUP_IDX_BYTES + rdna_compute::GL_GROUP_SCALE_BYTES),
            "MQ3G256GL blob size disagrees with GL_MQ3_GROUP_IDX_BYTES/GL_GROUP_SCALE_BYTES"
        );
    }

    #[test]
    pub(crate) fn e2m1_lookup_matches_ocp_spec() {
        // OCP MX FP4 (E2M1) spec values for the 8 magnitude codes.
        // Sign bit (0x8) flips sign of the magnitude.
        let expected: &[(u8, f32)] = &[
            (0x0, 0.0),
            (0x1, 0.5),
            (0x2, 1.0),
            (0x3, 1.5),
            (0x4, 2.0),
            (0x5, 3.0),
            (0x6, 4.0),
            (0x7, 6.0),
            (0x8, -0.0),
            (0x9, -0.5),
            (0xA, -1.0),
            (0xB, -1.5),
            (0xC, -2.0),
            (0xD, -3.0),
            (0xE, -4.0),
            (0xF, -6.0),
        ];
        for &(nib, want) in expected {
            assert_eq!(
                e2m1_to_f32(nib),
                want,
                "e2m1_to_f32(0x{:x}) = {} want {}",
                nib,
                e2m1_to_f32(nib),
                want
            );
        }
    }

    #[test]
    pub(crate) fn e2m1_dequant_unpacks_nibbles_and_doubles_logical_cols() {
        // Storage: 1 row × 1 col-byte. Byte = 0x42 → low nibble 0x2 (=1.0),
        // high nibble 0x4 (=2.0). Scale: 1 row × 1 col, UE8M0=127 (=2^0=1.0).
        // → logical row should be [1.0, 2.0] (length 2).
        let (vals, shape) = dequantize_e2m1_ue8m0_to_f32(&[0x42], &[1, 1], &[127], &[1, 1]);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    pub(crate) fn e2m1_dequant_applies_ue8m0_scale() {
        // Byte = 0x12 → low=2 (=1.0), high=1 (=0.5). Scale byte 128 → 2^1=2.0.
        // → logical [2.0, 1.0].
        let (vals, _) = dequantize_e2m1_ue8m0_to_f32(&[0x12], &[1, 1], &[128], &[1, 1]);
        assert_eq!(vals, vec![2.0, 1.0]);
    }

    #[test]
    pub(crate) fn parse_layer_idx_safetensors_dense() {
        assert_eq!(
            parse_layer_idx("model.layers.0.self_attn.q_proj.weight"),
            Some(0)
        );
        assert_eq!(
            parse_layer_idx("model.layers.63.mlp.gate_proj.weight"),
            Some(63)
        );
    }

    #[test]
    pub(crate) fn parse_layer_idx_safetensors_moe() {
        assert_eq!(
            parse_layer_idx("model.language_model.layers.5.mlp.experts.0.gate_up_proj.weight"),
            Some(5)
        );
    }

    #[test]
    pub(crate) fn parse_layer_idx_gguf() {
        assert_eq!(parse_layer_idx("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_idx("blk.31.ffn_gate.weight"), Some(31));
    }

    #[test]
    pub(crate) fn parse_layer_idx_no_match() {
        assert_eq!(parse_layer_idx("token_embd.weight"), None);
        assert_eq!(parse_layer_idx("output.weight"), None);
    }

    #[test]
    pub(crate) fn kmap_norms_are_f16() {
        assert_eq!(
            kmap_resolve("model.layers.0.input_layernorm.weight", 64, false),
            QuantLevel::F16
        );
        assert_eq!(
            kmap_resolve("model.layers.30.post_attention_layernorm.weight", 64, false),
            QuantLevel::F16
        );
    }

    #[test]
    pub(crate) fn kmap_embeds_are_q8() {
        assert_eq!(
            kmap_resolve("model.embed_tokens.weight", 64, false),
            QuantLevel::Q8
        );
        assert_eq!(kmap_resolve("lm_head.weight", 64, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve("output.weight", 64, false), QuantLevel::Q8);
    }

    #[test]
    pub(crate) fn kmap_moe_router_q8() {
        assert_eq!(
            kmap_resolve("model.language_model.layers.5.mlp.gate.weight", 64, true),
            QuantLevel::Q8
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.5.mlp.shared_expert_gate.weight",
                64,
                true
            ),
            QuantLevel::Q8
        );
    }

    #[test]
    pub(crate) fn kmap_moe_router_not_promoted_on_dense() {
        // On a dense model, mlp.gate.weight is not a router — falls to edge/base
        assert_ne!(
            kmap_resolve("model.layers.30.mlp.gate.weight", 64, false),
            QuantLevel::Q8
        );
    }

    #[test]
    pub(crate) fn kmap_moe_expert_ffn_promote6() {
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.30.mlp.experts.5.gate_up_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.30.mlp.experts.5.down_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
    }

    #[test]
    pub(crate) fn kmap_edge_layers_dense_ffn_only() {
        // Dense: FFN in edge layers — promoted
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.down_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.62.mlp.up_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.63.mlp.down_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        // Dense: attn in edge layers — NOT promoted
        assert_eq!(
            kmap_resolve("model.layers.0.self_attn.q_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.63.self_attn.v_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.0.linear_attn.in_proj_qkv.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn kmap_edge_layers_moe_attn_and_ffn() {
        // MoE: both attn and FFN in edge layers — promoted
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.63.self_attn.v_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
    }

    #[test]
    pub(crate) fn kmap_middle_layers_base() {
        assert_eq!(
            kmap_resolve("model.layers.2.self_attn.q_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.30.mlp.gate_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.61.mlp.down_proj.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn kmap_edge_layers_small_model_24_layers() {
        // 24 layers: edge = 0,1 and 22,23
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.2.mlp.gate_proj.weight", 24, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.22.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.23.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
    }

    #[test]
    pub(crate) fn kmap_n_layers_zero_disables_edge() {
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 0, false),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn kmap_edge_layers_tiny_model_3_layers() {
        // 3 layers: first-2 = {0,1}, last-2 = {1,2}. All layers promoted.
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.2.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
    }

    #[test]
    pub(crate) fn kmap_expert_not_promoted_on_dense() {
        // "mlp.experts." in name but is_moe=false — should NOT trigger rule 4
        assert_eq!(
            kmap_resolve(
                "model.layers.30.mlp.experts.5.gate_up_proj.weight",
                64,
                false
            ),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn kmap_gguf_names() {
        // GGUF edge-layer FFN (dense) — promoted
        assert_eq!(
            kmap_resolve("blk.0.ffn_gate.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("blk.63.ffn_gate.weight", 64, false),
            QuantLevel::Promote6
        );
        // GGUF edge-layer attn (dense) — NOT promoted
        assert_eq!(
            kmap_resolve("blk.0.attn_q.weight", 64, false),
            QuantLevel::Base
        );
        // GGUF edge-layer attn (MoE) — promoted
        assert_eq!(
            kmap_resolve("blk.0.attn_q.weight", 64, true),
            QuantLevel::Promote6
        );
        // GGUF middle-layer — base
        assert_eq!(
            kmap_resolve("blk.30.ffn_gate.weight", 64, false),
            QuantLevel::Base
        );
    }

    // ── Alternating mode tests ───────────────────────────────────────────

    #[test]
    pub(crate) fn positional_promote_edges() {
        assert!(is_positional_promote(0, 40, 3));
        assert!(is_positional_promote(1, 40, 3));
        assert!(is_positional_promote(38, 40, 3));
        assert!(is_positional_promote(39, 40, 3));
    }

    #[test]
    pub(crate) fn positional_promote_stride3() {
        // Middle layers: every 3rd starting from idx 2
        assert!(is_positional_promote(2, 40, 3)); // edge
        assert!(!is_positional_promote(3, 40, 3));
        assert!(!is_positional_promote(4, 40, 3));
        assert!(is_positional_promote(5, 40, 3));
        assert!(!is_positional_promote(6, 40, 3));
        assert!(!is_positional_promote(7, 40, 3));
        assert!(is_positional_promote(8, 40, 3));
    }

    #[test]
    pub(crate) fn kmap_alternating_moe_experts() {
        // MoE experts: promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.0.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Promote6 // edge layer
        );
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.5.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Promote6 // stride hit (5-2=3, 3%3==0)
        );
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.3.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Base // not on stride
        );
    }

    #[test]
    pub(crate) fn kmap_alternating_ffn_down() {
        // ffn_down promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // edge
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // stride
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.3.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Base // not on stride
        );
        // gate_proj NOT promoted in middle layers
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.gate_proj.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn kmap_alternating_n_layers_zero() {
        // With n_layers=0, alternating mode should return Base for everything
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 0, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn kmap_alternating_gguf_names() {
        // GGUF ffn_down in edge layer
        assert_eq!(
            kmap_resolve_mode("blk.0.ffn_down.weight", 40, false, 1),
            QuantLevel::Promote6
        );
        // GGUF ffn_down in middle non-stride layer
        assert_eq!(
            kmap_resolve_mode("blk.3.ffn_down.weight", 40, false, 1),
            QuantLevel::Base
        );
        // GGUF ffn_gate stays base in middle
        assert_eq!(
            kmap_resolve_mode("blk.5.ffn_gate.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn kmap_typed_promotes_down_and_v() {
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.down_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.15.self_attn.v_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        // gate_proj stays base
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.gate_proj.weight", 40, false, 2),
            QuantLevel::Base
        );
    }

    #[test]
    pub(crate) fn e8_soa_lsq_row_scale_does_not_increase_weight_mse() {
        let m = 2usize;
        let k = 256usize;
        let source: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.173).sin() * 1.7) + ((i % 11) as f32 - 5.0) * 0.031)
            .collect();
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let regular = quantize_mfp4g32_e8_soa_2d(&source, m, k, &s1, &s2);
        let repaired = quantize_mfp4g32_e8_soa_lsq_2d(&source, m, k, &s1, &s2);
        let q_regular = dequant_mfp4g32_e8_soa(&regular, m, k);
        let q_repaired = dequant_mfp4g32_e8_soa(&repaired, m, k);

        let mut rotated = source.clone();
        for row in rotated.chunks_mut(k) {
            cpu_fwht_256(row, &s1, &s2);
        }
        let mse = |q: &[f32]| {
            rotated
                .iter()
                .zip(q)
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f32>()
                / rotated.len() as f32
        };
        assert!(mse(&q_repaired) <= mse(&q_regular));
    }

    // ── Muse Glimmer (arch 14) classification locks ────────────────────────
    // These pin the new Glimmer tensor names to their intended quant decisions so a
    // future refactor cannot silently move them. Existing arches must not change.

    #[test]
    pub(crate) fn glimmer_lm_head_is_q8_separate_untied() {
        // Glimmer tie_word_embeddings=false — lm_head is SEPARATE from embed (unlike
        // Gemma4 which ties). Must land Q8 via Rule 2 (kmap_resolve_mode:50xx) and
        // via q8_class_of:is_q8_tensor (56xx). should_quantize keeps it quantizable
        // (contains "weight", not norm/bias, not vision).
        let name = "lm_head.weight";
        assert!(should_quantize(name), "lm_head must be quantizable (should_quantize:53xx)");
        assert_eq!(q8_class_of(name), Some("lm_head"), "q8_class_of:55xx lm_head");
        assert!(is_q8_tensor(name), "is_q8_tensor:59xx must be Q8");
        assert_eq!(kmap_resolve(name, 52, false), QuantLevel::Q8, "kmap Rule2 Q8");
        assert_eq!(kmap_resolve_mode(name, 52, false, 0), QuantLevel::Q8);
        assert_eq!(kmap_resolve_mode(name, 52, false, 3), QuantLevel::Q8);
    }

    #[test]
    pub(crate) fn glimmer_q8_classes_narrowing_keeps_lm_head_and_embed_only() {
        // The arch-14 default sets HIPFIRE_Q8_CLASSES=lm_head,embed and turns the
        // fixed tier on. This locks in what that narrowing actually selects: the
        // two output-side tensors are held at Q8, and attention is NOT — dragging
        // attention in would inflate the artifact for no stated requirement.
        //
        // Regression value: the first Glimmer MQ4 build shipped lm_head at
        // MQ4G256 even though `kmap_resolve` returns Q8 for it, because on a
        // dense model the fixed tier is off and the K-map verdict is never
        // consulted. The K-map assertions in the sibling test passed the entire
        // time. Only a check that pins the CLASS SELECTION catches that gap.
        //
        // SAFETY: single-threaded test; env is restored before returning.
        let prev = std::env::var("HIPFIRE_Q8_CLASSES").ok();
        unsafe { std::env::set_var("HIPFIRE_Q8_CLASSES", "lm_head,embed") };

        assert!(is_q8_tensor("lm_head.weight"), "lm_head must be Q8");
        assert!(
            is_q8_tensor("model.language_model.embed_tokens.weight"),
            "embed_tokens must be Q8"
        );
        let attn_q8 = is_q8_tensor("model.language_model.layers.0.self_attn.q_proj.weight");
        let gate_q8 = is_q8_tensor("model.language_model.layers.0.self_attn.gate_proj.weight");
        assert!(!attn_q8, "attention must NOT be pulled into Q8 by the glimmer default");
        assert!(!gate_q8, "the Glimmer attention gate is a projection and must follow --format");

        match prev {
            Some(v) => unsafe { std::env::set_var("HIPFIRE_Q8_CLASSES", v) },
            None => unsafe { std::env::remove_var("HIPFIRE_Q8_CLASSES") },
        }
    }

    #[test]
    pub(crate) fn glimmer_embed_tokens_is_q8() {
        let name = "model.language_model.embed_tokens.weight";
        assert!(should_quantize(name));
        assert_eq!(q8_class_of(name), Some("embed"));
        assert!(is_q8_tensor(name));
        assert_eq!(kmap_resolve(name, 52, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve_mode(name, 52, false, 1), QuantLevel::Q8);
    }

    #[test]
    pub(crate) fn glimmer_self_attn_gate_proj_is_attention_not_mlp_or_router() {
        // NEW name in Glimmer: self_attn.gate_proj gates attention output before
        // o_proj (see hipfire-arch-muse-glimmer lib.rs:24-27). Must be attn class,
        // not MLP gate and not MoE router.
        let attn_gate = "model.language_model.layers.0.self_attn.gate_proj.weight";
        let mlp_gate = "model.language_model.layers.0.mlp.gate_proj.weight";
        // q8_class_of:55xx — self_attn substring => "attn"; mlp gate has no
        // self_attn/attn_q/class and is not a router, so None.
        assert_eq!(q8_class_of(attn_gate), Some("attn"), "self_attn.gate_proj => attn (q8_class_of)");
        assert_eq!(q8_class_of(mlp_gate), None, "mlp.gate_proj must not be attn/router");
        assert!(is_q8_tensor(attn_gate), "attn gate must be fixed-tier Q8");
        assert!(!is_q8_tensor(mlp_gate), "mlp gate is not fixed-tier (unless --q8-router on MoE)");
        // should_quantize:53xx — both are weights, not norms/bias/vision => true
        assert!(should_quantize(attn_gate));
        assert!(should_quantize(mlp_gate));
        // awq_eligible:61xx — ends_with("gate_proj.weight") catches BOTH. This is
        // CORRECT for the attention gate: it is an input-side projection from the
        // normed hidden (post_input_layernorm) whose runtime path is the same
        // fused_rmsnorm_rotate AWQ kernel as mlp gate/up. The scale s[j] multiplies
        // the gate's input channels and is divided at inference before the gate;
        // the gate's output then scales attn_out via sigmoid. Input-side AWQ is
        // mathematically valid regardless of where the gate's output is applied.
        assert!(awq_eligible(attn_gate), "attn gate must be AWQ-eligible (input-side)");
        assert!(awq_eligible(mlp_gate), "mlp gate must be AWQ-eligible");
        // kmap: dense edge-layer rule promotes FFN only, not attn — so even in
        // edge layer 0, the attn gate stays Base (not Promote6). This matches the
        // dense policy (attn promotion regresses PPL +3.1%).
        assert_eq!(kmap_resolve(attn_gate, 52, false), QuantLevel::Base);
        // MoE is irrelevant for Glimmer (dense), but verify router rule does not
        // mis-fire even if is_moe were true: attn gate is not mlp.gate.weight.
        // For MoE edge-layer (0 is edge), full promotion returns Promote6 for every
        // tensor including attn — that is the expected MoE policy, not a router.
        assert_eq!(kmap_resolve_mode("model.layers.0.self_attn.gate_proj.weight", 52, true, 0), QuantLevel::Promote6);
    }

    #[test]
    pub(crate) fn glimmer_norms_are_f16_never_lowbit() {
        // Sandwich RMSNorms: input, post_attn, pre_ffn, post_ffn + final norm.
        // All contain "norm" => Rule1 F16 (kmap_resolve_mode:50xx) and
        // should_quantize:false (53xx). Must never be quantized.
        let norms = [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.0.pre_feedforward_layernorm.weight",
            "model.language_model.layers.0.post_feedforward_layernorm.weight",
            "model.language_model.layers.51.input_layernorm.weight",
            "model.language_model.norm.weight",
        ];
        for name in norms {
            assert!(!should_quantize(name), "norm {name} must not be quantizable");
            assert_eq!(kmap_resolve(name, 52, false), QuantLevel::F16, "kmap F16 for {name}");
            assert_eq!(kmap_resolve_mode(name, 52, false, 1), QuantLevel::F16);
            assert_eq!(kmap_resolve_mode(name, 52, false, 2), QuantLevel::F16);
            assert_eq!(kmap_resolve_mode(name, 52, false, 3), QuantLevel::F16);
            assert_eq!(kmap_resolve(name, 52, true), QuantLevel::F16, "even MoE must be F16");
            // q8_class_of is unrelated to norms — must be None / not Q8
            assert!(!is_q8_tensor(name));
        }
    }

    #[test]
    pub(crate) fn glimmer_vision_prefixes_are_f16_and_not_parsed_as_text_layers() {
        // Glimmer vision tensors are model.vision_tower.*, model.vision_adapter.*,
        // model.vision_projection.* (809 total). Previously is_vision/should_quantize
        // only matched model.visual./visual./vision_tower. — so
        // model.vision_tower.* fell through to text quant. Now extended additively
        // (should_quantize:53xx, main is_vision, parse_layer_idx:49xx).
        let vision = [
            "model.vision_tower.layers.0.attn.q_proj.weight",
            "model.vision_tower.layers.49.mlp.fc2.weight",
            "model.vision_tower.layers.25.norm1.weight",
            "model.vision_adapter.fc1.weight",
            "model.vision_projection.weight",
        ];
        for name in vision {
            assert!(!should_quantize(name), "vision {name} must stay F16 (should_quantize)");
            assert_eq!(kmap_resolve(name, 52, false), QuantLevel::F16, "kmap vision F16 for {name}");
            assert_eq!(kmap_resolve_mode(name, 52, false, 0), QuantLevel::F16);
            assert_eq!(kmap_resolve_mode(name, 52, true, 1), QuantLevel::F16);
            // parse_layer_idx must NOT extract vision_tower.layers.N as text layer
            assert_eq!(parse_layer_idx(name), None, "vision {name} must not parse as layer idx");
            // The old unanchored find("layers.") would have returned Some(0/49)
            // and edge-layer Promote6 could have fired — locked to None now.
        }
        // Plain vision_tower. prefix (dots.ocr style) must still be F16
        assert_eq!(kmap_resolve("vision_tower.layers.0.attn.q_proj.weight", 52, false), QuantLevel::F16);
        assert_eq!(parse_layer_idx("vision_tower.layers.0.attn.q_proj.weight"), None);
        // model.visual.* (Qwen3.5-VL) unchanged
        assert!(!should_quantize("model.visual.patch_embed.weight"));
        assert_eq!(parse_layer_idx("model.visual.layers.0.weight"), None);
    }

    #[test]
    pub(crate) fn glimmer_text_layers_still_parse() {
        // Sanity: text layers must still parse correctly (no regression for non-vision).
        assert_eq!(parse_layer_idx("model.language_model.layers.0.self_attn.q_proj.weight"), Some(0));
        assert_eq!(parse_layer_idx("model.language_model.layers.51.mlp.down_proj.weight"), Some(51));
        assert_eq!(parse_layer_idx("model.layers.3.self_attn.gate_proj.weight"), Some(3));
    }

    #[test]
    pub(crate) fn e8_soa_awls_row_scale_does_not_increase_weighted_mse() {
        let m = 2usize;
        let k = 256usize;
        let source: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.119).cos() * 1.3) + ((i % 17) as f32 - 8.0) * 0.023)
            .collect();
        let importance: Vec<f64> = (0..k)
            .map(|i| 0.05 + ((i * 37 % 101) as f64 / 17.0))
            .collect();
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let regular = quantize_mfp4g32_e8_soa_2d(&source, m, k, &s1, &s2);
        let repaired = quantize_mfp4g32_e8_soa_awls_2d(&source, m, k, &s1, &s2, &importance);
        let q_regular = dequant_mfp4g32_e8_soa(&regular, m, k);
        let q_repaired = dequant_mfp4g32_e8_soa(&repaired, m, k);

        let mut rotated = source.clone();
        for row in rotated.chunks_mut(k) {
            cpu_fwht_256(row, &s1, &s2);
        }
        let weighted_mse = |q: &[f32]| {
            rotated
                .iter()
                .zip(q)
                .enumerate()
                .map(|(i, (a, b))| {
                    let d = (*a - *b) as f64;
                    importance[i % k] * d * d
                })
                .sum::<f64>()
                / (m * k) as f64
        };
        assert!(weighted_mse(&q_repaired) <= weighted_mse(&q_regular));
    }
}
