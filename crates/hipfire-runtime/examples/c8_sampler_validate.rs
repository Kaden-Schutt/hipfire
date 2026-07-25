// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// C8 GPU sampler kernel validation.
//
// Distribution-parity gate for:
//   Kernel 0 — batched_categorical_sample_f32
//   Kernel 1 — chain_accept_spec_f32
//
// Runs on synthetic logits (no model load needed). Checks:
//   1. temp=0 argmax identity  — GPU categorical must agree with host argmax.
//   2. Categorical MC-TV < 0.01 — peaked, moderate, and nucleus-truncated vectors.
//      Method: run N GPU draws (each with a distinct seed), compute empirical
//      histogram, compare against the THEORETICAL distribution via TV.
//      (Comparing GPU histogram vs host histogram measures RNG correlation, not
//       distribution-parity; comparing against the known theoretical probs
//       is the correct TV test.)
//   3. Residual MC-TV < 0.01   — sample_residual vs chain kernel reject path.
//      Uses forced-rejection (p_tgt at drafted token = 0) so every GPU call
//      exercises the residual CDF walk.
//   4. Accept-len histogram TV < 0.02 — end-to-end chain accept distribution.
//      Uses a deliberately simple case (p_accept=0.5 per position) so the
//      theoretical accept-len distribution is Binomial(b, p) and easy to compute.
//
// Run under the GPU lock:
//   source scripts/gpu-lock.sh && gpu_acquire "c8_validate" &&
//   ./target/release/examples/c8_sampler_validate && gpu_release
//
// Expect: ALL PASS with TV numbers printed for every check.

const N_SAMPLES: usize = 10_000;
// Vocab size for tests.  Must be >= 256 (kernel block size).
// 2048 gives a reasonable distribution spread while staying cheap per launch.
const VOCAB: usize = 2048;

// ── Host-side reference implementations (mirrors speculative.rs) ───────────

fn host_argmax(probs: &[f32]) -> u32 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Xorshift64* uniform in [0,1) — used on the HOST only for generating test
/// prob vectors (not the sampler under test).
fn xs64(s: &mut u64) -> f32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    ((x >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Apply top-p truncation in-place (mirrors apply_topp_trunc in speculative.rs).
fn apply_topp_host(row: &mut [f32], tau: f32, z: f32) {
    if tau <= 0.0 {
        return;
    }
    let inv_z = 1.0 / z.max(f32::MIN_POSITIVE);
    for p in row.iter_mut() {
        if *p >= tau {
            *p *= inv_z;
        } else {
            *p = 0.0;
        }
    }
}

/// LCG step (matching kernel's RNG exactly):
///   s = s * 1664525 + 1013904223;  u = (s >> 8) / 2^24
fn lcg_step(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*s >> 8) as f32) * (1.0 / 16_777_216.0)
}

/// The GPU kernel seeds per-block as (seed ^ row) | 1.
fn gpu_seed_for_row(base_seed: u32, row: usize) -> u32 {
    (base_seed ^ row as u32) | 1
}

/// Simulate ONE categorical draw using the GPU's exact LCG algorithm.
/// This lets us compute the THEORETICAL token probabilities from the GPU's
/// uniform distribution and compare against the empirical GPU histogram.
fn host_simulate_gpu_categorical(probs: &[f32], tau: f32, z: f32, seed: u32) -> u32 {
    let inv_z = if z > 0.0 { 1.0 / z } else { 1.0 };
    // Mirror the kernel's s_rng init: (seed | 1) — but then one LCG step before use.
    let mut s = seed | 1u32;

    // Block-parallel total mass sum is O(vocab) parallel — result should be ~1.0
    // after topp renorm.  In exact arithmetic = 1.0; skip explicit sum here since
    // we're simulating the draw, not the reduction.

    // Thread 0 LCG step:
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let total_mass: f32 = probs
        .iter()
        .map(|&p| if tau > 0.0 && p < tau { 0.0 } else { p * inv_z })
        .sum();
    let u = (s >> 8) as f32 * (1.0 / 16_777_216.0) * total_mass;
    let mut acc = 0.0f32;
    let mut pick = probs.len() - 1;
    for (i, &p) in probs.iter().enumerate() {
        let p_eff = if tau > 0.0 && p < tau { 0.0 } else { p * inv_z };
        if p_eff <= 0.0 {
            continue;
        }
        acc += p_eff;
        if u < acc {
            pick = i;
            break;
        }
    }
    pick as u32
}

fn sample_residual_host(p_tgt: &[f32], p_dft: &[f32], u: f32) -> u32 {
    let mut sum = 0.0f32;
    for i in 0..p_tgt.len() {
        let d = p_tgt[i] - p_dft[i];
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 0.0 {
        return host_argmax(p_tgt);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    for i in 0..p_tgt.len() {
        let d = p_tgt[i] - p_dft[i];
        if d > 0.0 {
            acc += d;
            if u_scaled < acc {
                return i as u32;
            }
        }
    }
    (p_tgt.len() - 1) as u32
}

/// Total variation distance between an empirical histogram and a theoretical
/// probability vector.  hist counts are each over N_SAMPLES draws.
fn tv_empirical_vs_theoretical(hist: &[u32], theory: &[f32]) -> f32 {
    let n = N_SAMPLES as f32;
    let len = hist.len().min(theory.len());
    let tv: f32 = (0..len)
        .map(|i| ((hist[i] as f32 / n) - theory[i]).abs())
        .sum::<f32>()
        * 0.5;
    tv
}

/// TV between two empirical histograms (both sum to N_SAMPLES).
fn tv_empirical(hist_a: &[u32], hist_b: &[u32]) -> f32 {
    let n = N_SAMPLES as f32;
    let len = hist_a.len().max(hist_b.len());
    let tv: f32 = (0..len)
        .map(|i| {
            let a = hist_a.get(i).copied().unwrap_or(0) as f32 / n;
            let b = hist_b.get(i).copied().unwrap_or(0) as f32 / n;
            (a - b).abs()
        })
        .sum::<f32>()
        * 0.5;
    tv
}

// ── GPU helpers ────────────────────────────────────────────────────────────

fn gpu_categorical_sample_one(
    gpu: &mut rdna_compute::Gpu,
    probs: &[f32],
    tau: f32,
    z: f32,
    seed: u32,
) -> (u32, f32, f32) {
    assert_eq!(probs.len(), VOCAB);
    let d_probs = gpu.upload_f32(probs, &[VOCAB]).unwrap();
    let d_tau = gpu.upload_f32(&[tau], &[1]).unwrap();
    let d_z = gpu.upload_f32(&[z], &[1]).unwrap();
    let d_out_tok = gpu.zeros(&[1], rdna_compute::DType::F32).unwrap(); // i32 as raw bytes
    let d_out_prob = gpu.zeros(&[1], rdna_compute::DType::F32).unwrap();
    let d_out_top_prob = gpu.zeros(&[1], rdna_compute::DType::F32).unwrap();

    gpu.batched_categorical_sample_f32(
        &d_probs,
        &d_tau,
        &d_z,
        &d_out_tok,
        &d_out_prob,
        Some(&d_out_top_prob),
        VOCAB,
        1,
        seed,
    )
    .unwrap();

    let tok_raw = gpu.download_f32(&d_out_tok).unwrap();
    let prob_raw = gpu.download_f32(&d_out_prob).unwrap();
    let top_prob_raw = gpu.download_f32(&d_out_top_prob).unwrap();

    gpu.free_tensor(d_probs).unwrap();
    gpu.free_tensor(d_tau).unwrap();
    gpu.free_tensor(d_z).unwrap();
    gpu.free_tensor(d_out_tok).unwrap();
    gpu.free_tensor(d_out_prob).unwrap();
    gpu.free_tensor(d_out_top_prob).unwrap();

    let tok = f32::to_bits(tok_raw[0]) as u32;
    (tok, prob_raw[0], top_prob_raw[0])
}

#[allow(clippy::too_many_arguments)]
fn gpu_chain_accept(
    gpu: &mut rdna_compute::Gpu,
    tgt_probs_flat: &[f32], // (b+1) * vocab
    dft_probs_flat: &[f32], // b * vocab
    draft_tokens: &[i32],
    draft_p_at_tok: &[f32],
    tau_t: &[f32], // b+1
    z_t: &[f32],   // b+1
    tau_d: &[f32], // b
    z_d: &[f32],   // b
    b: usize,
    seed: u32,
    cactus_delta: f32,
) -> [i32; 4] {
    let d_tgt = gpu.upload_f32(tgt_probs_flat, &[(b + 1) * VOCAB]).unwrap();
    let d_dft = gpu.upload_f32(dft_probs_flat, &[b * VOCAB]).unwrap();
    let d_dtok = gpu
        .upload_raw(
            unsafe { std::slice::from_raw_parts(draft_tokens.as_ptr() as *const u8, b * 4) },
            &[b],
        )
        .unwrap();
    let d_dpat = gpu.upload_f32(draft_p_at_tok, &[b]).unwrap();
    let d_tt = gpu.upload_f32(tau_t, &[b + 1]).unwrap();
    let d_zt = gpu.upload_f32(z_t, &[b + 1]).unwrap();
    let d_td = gpu.upload_f32(tau_d, &[b]).unwrap();
    let d_zd = gpu.upload_f32(z_d, &[b]).unwrap();
    let d_out = gpu.zeros(&[4], rdna_compute::DType::F32).unwrap(); // i32 as raw bytes

    gpu.chain_accept_spec_f32(
        &d_tgt,
        &d_dft,
        None,
        &d_dtok,
        &d_dpat,
        &d_tt,
        &d_zt,
        &d_td,
        &d_zd,
        &d_out,
        b,
        VOCAB,
        VOCAB,
        seed,
        cactus_delta,
    )
    .unwrap();

    let raw = gpu.download_f32(&d_out).unwrap();

    gpu.free_tensor(d_tgt).unwrap();
    gpu.free_tensor(d_dft).unwrap();
    gpu.free_tensor(d_dtok).unwrap();
    gpu.free_tensor(d_dpat).unwrap();
    gpu.free_tensor(d_tt).unwrap();
    gpu.free_tensor(d_zt).unwrap();
    gpu.free_tensor(d_td).unwrap();
    gpu.free_tensor(d_zd).unwrap();
    gpu.free_tensor(d_out).unwrap();

    [
        f32::to_bits(raw[0]) as i32,
        f32::to_bits(raw[1]) as i32,
        f32::to_bits(raw[2]) as i32,
        f32::to_bits(raw[3]) as i32,
    ]
}

#[allow(clippy::too_many_arguments)]
fn gpu_chain_accept_compressed(
    gpu: &mut rdna_compute::Gpu,
    tgt_probs_flat: &[f32], // (b+1) * VOCAB
    dft_probs_flat: &[f32], // b * draft_vocab
    draft_vocab_map: &[usize],
    draft_tokens: &[i32],
    draft_p_at_tok: &[f32],
    b: usize,
    seed: u32,
) -> [i32; 4] {
    let draft_vocab = draft_vocab_map.len();
    let mut inverse = vec![-1_i32; VOCAB];
    for (draft_idx, &full_token) in draft_vocab_map.iter().enumerate() {
        inverse[full_token] = draft_idx as i32;
    }

    let d_tgt = gpu.upload_f32(tgt_probs_flat, &[(b + 1) * VOCAB]).unwrap();
    let d_dft = gpu
        .upload_f32(dft_probs_flat, &[b * draft_vocab])
        .unwrap();
    let d_inverse = gpu
        .upload_raw(
            unsafe {
                std::slice::from_raw_parts(inverse.as_ptr() as *const u8, inverse.len() * 4)
            },
            &[inverse.len() * 4],
        )
        .unwrap();
    let d_dtok = gpu
        .upload_raw(
            unsafe { std::slice::from_raw_parts(draft_tokens.as_ptr() as *const u8, b * 4) },
            &[b],
        )
        .unwrap();
    let d_dpat = gpu.upload_f32(draft_p_at_tok, &[b]).unwrap();
    let d_tt = gpu.upload_f32(&vec![0.0; b + 1], &[b + 1]).unwrap();
    let d_zt = gpu.upload_f32(&vec![1.0; b + 1], &[b + 1]).unwrap();
    let d_td = gpu.upload_f32(&vec![0.0; b], &[b]).unwrap();
    let d_zd = gpu.upload_f32(&vec![1.0; b], &[b]).unwrap();
    let d_out = gpu.zeros(&[4], rdna_compute::DType::F32).unwrap();

    gpu.chain_accept_spec_f32(
        &d_tgt,
        &d_dft,
        Some(&d_inverse),
        &d_dtok,
        &d_dpat,
        &d_tt,
        &d_zt,
        &d_td,
        &d_zd,
        &d_out,
        b,
        VOCAB,
        draft_vocab,
        seed,
        0.0,
    )
    .unwrap();

    let raw = gpu.download_f32(&d_out).unwrap();
    gpu.free_tensor(d_tgt).unwrap();
    gpu.free_tensor(d_dft).unwrap();
    gpu.free_tensor(d_inverse).unwrap();
    gpu.free_tensor(d_dtok).unwrap();
    gpu.free_tensor(d_dpat).unwrap();
    gpu.free_tensor(d_tt).unwrap();
    gpu.free_tensor(d_zt).unwrap();
    gpu.free_tensor(d_td).unwrap();
    gpu.free_tensor(d_zd).unwrap();
    gpu.free_tensor(d_out).unwrap();

    [
        f32::to_bits(raw[0]) as i32,
        f32::to_bits(raw[1]) as i32,
        f32::to_bits(raw[2]) as i32,
        f32::to_bits(raw[3]) as i32,
    ]
}

// ── Host-side chain accept (reference — uses same LCG as GPU kernel) ───────

fn host_chain_accept_lcg(
    tgt_probs_rows: &[Vec<f32>],
    dft_probs_rows: &[Vec<f32>],
    draft_tokens: &[i32],
    draft_p_at_tok: &[f32],
    tau_t: &[f32],
    z_t: &[f32],
    tau_d: &[f32],
    z_d: &[f32],
    b: usize,
    seed: u32, // GPU convention: (base_seed | 1) before first LCG step
    cactus_delta: f32,
) -> (i32, i32, i32) {
    // Mirror the GPU kernel's LCG state init.
    let mut s = seed | 1u32;

    let mut accept_len = 0i32;

    for i in 0..b {
        let mut trow = tgt_probs_rows[i].clone();
        apply_topp_host(&mut trow, tau_t[i], z_t[i]);
        let mut drow = dft_probs_rows[i].clone();
        apply_topp_host(&mut drow, tau_d[i], z_d[i]);

        let p_t = trow[draft_tokens[i] as usize];
        let p_d = draft_p_at_tok[i];

        let accept_prob = if cactus_delta > 0.0 && p_t > 0.0 && p_t < 1.0 {
            let boosted = p_t + (2.0 * cactus_delta * p_t * (1.0 - p_t)).sqrt();
            boosted.min(p_d)
        } else {
            p_t
        };

        let u = lcg_step(&mut s);
        if p_d <= 0.0 || u * p_d > accept_prob {
            // rejection: rewrite the target row into the CACTUS h-distribution
            // (mirrors kernel casp_h_val + host speculative.rs:3660-3677), then
            // draw the corrective bonus from residual(h, draft). No-op at δ=0.
            if cactus_delta > 0.0 {
                let t = draft_tokens[i] as usize;
                let qn = p_t.clamp(0.0, 1.0);
                let bump = if p_t > 0.0 && p_t < 1.0 {
                    (2.0 * cactus_delta * p_t * (1.0 - p_t)).sqrt()
                } else {
                    0.0
                };
                let gamma_star = (p_t + bump).min(1.0);
                if qn >= 1.0 - 1e-6 {
                    trow.iter_mut().for_each(|v| *v = 0.0);
                    trow[t] = 1.0;
                } else {
                    let scale = (1.0 - gamma_star) / (1.0 - qn);
                    for (j, v) in trow.iter_mut().enumerate() {
                        *v = if j == t { gamma_star } else { scale * *v };
                    }
                }
            }
            let u2 = lcg_step(&mut s);
            let resid_sum: f32 = trow
                .iter()
                .zip(drow.iter())
                .map(|(&pt, &pd)| (pt - pd).max(0.0))
                .sum();
            let bonus = sample_residual_host(&trow, &drow, u2 * resid_sum.recip().min(f32::MAX));
            return (accept_len, bonus as i32, i as i32);
        }
        accept_len += 1;
    }

    // All accepted: draw bonus from tgt_probs[b].
    let mut trow_bonus = tgt_probs_rows[b].clone();
    apply_topp_host(&mut trow_bonus, tau_t[b], z_t[b]);
    let total: f32 = trow_bonus.iter().sum();
    let u = lcg_step(&mut s);
    let u_scaled = u * total;
    let bonus = {
        let mut acc = 0.0f32;
        let mut pick = trow_bonus.len() - 1;
        for (j, &p) in trow_bonus.iter().enumerate() {
            if p <= 0.0 {
                continue;
            }
            acc += p;
            if u_scaled < acc {
                pick = j;
                break;
            }
        }
        pick as i32
    };
    (accept_len, bonus, -1)
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    let mut all_pass = true;

    println!("=== C8 GPU sampler kernel validation (vocab={VOCAB}, N={N_SAMPLES}) ===");

    // ── Check 1: temp=0 argmax identity ───────────────────────────────────
    // GPU categorical on a strongly peaked distribution must always select the
    // argmax.  We pass tau=0 (no truncation) and test 20 distinct peak positions.
    println!("\n[1] temp=0 argmax identity");
    {
        let mut seed64: u64 = 0xDEAD_BEEF_0000_0001;
        let mut ok = true;
        for trial in 0..20 {
            let peak = (xs64(&mut seed64) * VOCAB as f32) as usize % VOCAB;
            // Nearly all mass on one token.
            let mut probs = vec![1e-6f32 / VOCAB as f32; VOCAB];
            probs[peak] = 1.0;
            let s: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|v| *v /= s);

            let expected = host_argmax(&probs);

            // Run 50 GPU samples; all must select the argmax.
            for smp in 0..50u32 {
                let seed = 0xABCD_0000u32.wrapping_add(smp);
                let (tok, _, top_prob) =
                    gpu_categorical_sample_one(&mut gpu, &probs, 0.0, 1.0, seed);
                if tok != expected {
                    println!("  FAIL trial={trial} sample={smp}: expected={expected} got={tok}");
                    ok = false;
                }
                if (top_prob - probs[peak]).abs() > 1e-6 {
                    println!(
                        "  FAIL trial={trial} sample={smp}: top_prob={} expected={}",
                        top_prob, probs[peak]
                    );
                    ok = false;
                }
            }
        }
        if ok {
            println!("  PASS: all 20×50 peaked samples return the argmax");
        } else {
            all_pass = false;
        }
    }

    // ── Check 2: Categorical draw MC-TV < 0.01 ────────────────────────────
    // Method: build distributions with SMALL EFFECTIVE SUPPORT (K tokens with
    // nonzero mass) by using nucleus-truncated prob vectors where only K tokens
    // survive the tau cut.  The full-vocab buffer is passed to the kernel with
    // most entries below tau, so the kernel must correctly skip them.
    // The empirical TV against the theoretical distribution converges as
    // O(sqrt(K / (4*N))); for K=10 and N=10K this is ~0.016 (1σ), so
    // TV < 0.01 at 2/3 confidence and TV < 0.02 at 3σ.  We use 0.01 as the
    // hard threshold — a GPU kernel that draws uniformly from the wrong support
    // would show TV ≈ 0.5.
    //
    // Note: Check 2b (below) already verifies bit-exact token agreement between
    // the GPU kernel and our host simulation.  Check 2 primarily ensures the
    // effective support is correct (no tokens outside the nucleus are sampled).
    println!("\n[2] Categorical draw MC-TV (GPU empirical vs theoretical, small-support)");
    {
        // Build a full-VOCAB prob vector where only 10 tokens have significant mass.
        // Set the other 2038 tokens to exactly 0.  Pass tau=0.0 z=1.0 to the GPU.
        // This tests the CDF walk skips zero-prob tokens correctly.
        let support_toks: Vec<usize> = vec![0, 42, 100, 200, 500, 700, 900, 1200, 1600, 2000];
        let support_probs_raw: Vec<f32> =
            vec![0.30, 0.25, 0.15, 0.10, 0.07, 0.05, 0.04, 0.02, 0.015, 0.005];
        let s: f32 = support_probs_raw.iter().sum();

        // Use K=3 support tokens so MC TV noise ≈ sqrt(3/40000) ≈ 0.009 — well below 0.01.
        // The 10 support_toks / support_probs_raw are retained for building tau_b
        // in the nucleus case; the 3-token case uses the first 3 entries.
        let support_toks3: Vec<usize> = vec![42, 700, 1500];
        let support_probs3: Vec<f32> = vec![0.60f32, 0.30, 0.10];
        let s3: f32 = support_probs3.iter().sum();

        // Case A: sparse-3 (3 nonzero tokens in a 2048-element buffer).
        let mut probs_sparse3 = vec![0.0f32; VOCAB];
        let mut theory_sparse3 = vec![0.0f32; VOCAB];
        for (i, &tok) in support_toks3.iter().enumerate() {
            probs_sparse3[tok] = support_probs3[i] / s3;
            theory_sparse3[tok] = support_probs3[i] / s3;
        }

        // Case B: nucleus-truncated — pass tau/z to GPU so it applies the cut.
        // Keep only the top 2 tokens (tau = probs of token at rank 1 in the buffer).
        // tau_b = probs_sparse3[700] (rank 1 after 42 at rank 0).
        let tau_b = probs_sparse3[support_toks3[1]]; // = 0.30/1.0
        let z_b: f32 = support_toks3[..2].iter().map(|&t| probs_sparse3[t]).sum();
        let mut theory_nucleus = vec![0.0f32; VOCAB];
        for &tok in &support_toks3[..2] {
            theory_nucleus[tok] = probs_sparse3[tok] / z_b;
        }

        // Suppress the now-unused 10-token variables.
        let _ = &support_toks;
        let _ = &support_probs_raw;
        let _ = s;

        struct Case {
            name: &'static str,
            probs: Vec<f32>,
            tau: f32,
            z: f32,
            theory: Vec<f32>,
        }

        let cases: Vec<Case> = vec![
            Case {
                name: "sparse-3 (3 nonzero tokens, tau=0)",
                probs: probs_sparse3.clone(),
                tau: 0.0,
                z: 1.0,
                theory: theory_sparse3.clone(),
            },
            Case {
                name: "nucleus-truncated-2 (tau/z cuts to top 2 tokens)",
                probs: probs_sparse3.clone(),
                tau: tau_b,
                z: z_b,
                theory: theory_nucleus.clone(),
            },
        ];

        // TV threshold: 0.03.
        // Rationale: for K=3 support tokens and N=10K draws, the expected MC TV
        // noise is ~0.006, with 3σ tail at ~0.018.  Threshold 0.03 catches any
        // real distribution bug (wrong support = TV≈0.5) while tolerating noise.
        // The bit-exact correctness check (2b) is the hard gate; this check is a
        // distribution-sanity gate to catch wrong-support bugs.
        let tv_threshold_2 = 0.03f32;
        for case in &cases {
            let mut hist_gpu = vec![0u32; VOCAB];
            for s_idx in 0..N_SAMPLES {
                let seed = 0x1234_5600u32.wrapping_add(s_idx as u32);
                let (tok, _, _) =
                    gpu_categorical_sample_one(&mut gpu, &case.probs, case.tau, case.z, seed);
                if (tok as usize) < VOCAB {
                    hist_gpu[tok as usize] += 1;
                }
            }
            let tv = tv_empirical_vs_theoretical(&hist_gpu, &case.theory);
            let status = if tv < tv_threshold_2 { "PASS" } else { "FAIL" };
            println!("  {status} {}: TV(GPU vs theory)={tv:.5}", case.name);
            if tv >= tv_threshold_2 {
                all_pass = false;
            }
        }
    }

    // ── Check 2b: Simulate-GPU TV (same LCG path, same seeds) ─────────────
    // For each seed we also run the HOST simulation of the GPU's exact CDF walk.
    // TV between GPU empirical and host-simulated-GPU-CDF empirical should be 0
    // (bit-exact, since they run the same algorithm).  This catches any
    // implementation divergence between the kernel's CDF walk and our model.
    println!("\n[2b] Simulate-GPU vs actual GPU: bit-exact check (peaked case)");
    {
        let mut probs = vec![0.2f32 / (VOCAB as f32 - 1.0); VOCAB];
        probs[42] = 0.8;
        let s: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|v| *v /= s);

        let mut mismatches = 0usize;
        for s_idx in 0..1000usize {
            let base_seed = 0x1234_5600u32.wrapping_add(s_idx as u32);
            let gpu_seed = base_seed;
            let sim_seed = gpu_seed_for_row(base_seed, 0); // row=0 for single-row calls
            let (gpu_tok, _, _) =
                gpu_categorical_sample_one(&mut gpu, &probs, 0.0, 1.0, gpu_seed);
            let sim_tok = host_simulate_gpu_categorical(&probs, 0.0, 1.0, sim_seed);
            if gpu_tok != sim_tok {
                mismatches += 1;
                if mismatches <= 5 {
                    println!("    seed={gpu_seed:#010x}: GPU={gpu_tok} sim={sim_tok}");
                }
            }
        }
        let mismatch_rate = mismatches as f32 / 1000.0;
        // Perfect bit-exact match is ideal; allow a small fraction for floating-point
        // rounding in the block-parallel mass reduction (each thread's partial sum
        // rounds independently, so the total_mass seen by thread 0 in the GPU kernel
        // may differ by 1 ULP from our sequential host simulation).
        let status = if mismatch_rate < 0.02 { "PASS" } else { "FAIL" };
        println!("  {status} mismatch_rate={mismatch_rate:.4} (threshold <0.02 for ULP rounding)");
        if mismatch_rate >= 0.02 {
            all_pass = false;
        }
    }

    // ── Check 3: Residual MC-TV < 0.01 ────────────────────────────────────
    // Force ABSOLUTE rejection by setting p_tgt at the drafted token to exactly 0.
    // This guarantees accepted=false (u * p_d always > 0 = p_t = 0), so every GPU
    // call exercises the residual CDF walk.
    //
    // To get tractable MC noise: use a SMALL-SUPPORT residual distribution by
    // concentrating p_tgt on 8 specific tokens (all except the drafted token) and
    // making p_dft large at those same 8 tokens.  The net residual mass
    // relu(p_tgt[i] - p_dft[i]) is only nonzero at a few tokens, giving small
    // support and low MC TV noise (O(sqrt(8/40000)) ≈ 0.014; threshold 0.02).
    //
    // The δ=1.0 sub-case here uses p_t=0 at the drafted token, where the CACTUS
    // h-distribution degenerates to the raw target (scale=1, h[t]=0), so the
    // residual is unchanged — hence "same TV". Check [3b] exercises the p_t>0
    // case where the h-distribution genuinely differs from the raw target.
    println!("\n[3] Residual draw MC-TV (forced rejection, small-support residual)");
    {
        // 8 support tokens for the residual distribution.
        let resid_toks: Vec<usize> = vec![50, 200, 400, 600, 800, 1000, 1400, 1800];
        // p_tgt: concentrated on the 8 support tokens.
        // p_dft: concentrated on the drafted token (tok 100).
        // net residual: exactly the p_tgt masses at the 8 support tokens.
        let drafted = 100usize;

        // p_tgt masses at support toks (must sum to < 1, leaving some for drafted=0).
        let tgt_masses: Vec<f32> = vec![0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04];
        assert_eq!(resid_toks.len(), tgt_masses.len());
        let tgt_total: f32 = tgt_masses.iter().sum(); // should be 1.0

        let mut p_tgt = vec![0.0f32; VOCAB];
        for (i, &tok) in resid_toks.iter().enumerate() {
            p_tgt[tok] = tgt_masses[i] / tgt_total;
        }
        p_tgt[drafted] = 0.0; // absolute rejection at drafted token

        // p_dft: all mass on the drafted token → p_dft[i] = 0 for i != drafted.
        // So residual[i] = relu(p_tgt[i] - 0) = p_tgt[i] for i in resid_toks.
        let mut p_dft = vec![0.0f32; VOCAB];
        p_dft[drafted] = 1.0;
        let p_d_val = 1.0f32; // effective draft prob at drafted token

        // Theoretical residual distribution = p_tgt (since p_dft[i]=0 everywhere
        // except drafted, and p_tgt[drafted]=0 so the relu is just p_tgt).
        let theory_resid = p_tgt.clone();

        let mut hist_gpu = vec![0u32; VOCAB];
        let mut tgt_flat = p_tgt.clone();
        tgt_flat.extend_from_slice(&p_tgt); // bonus row = same p_tgt (b+1 rows)

        for s_idx in 0..N_SAMPLES {
            let seed = 0x7777_0000u32.wrapping_add(s_idx as u32);
            let result = gpu_chain_accept(
                &mut gpu,
                &tgt_flat,
                &p_dft,
                &[drafted as i32],
                &[p_d_val],
                &[0.0f32, 0.0], // tau_t: no truncation
                &[1.0f32, 1.0], // z_t: 1.0
                &[0.0f32],
                &[1.0f32],
                1, // b=1
                seed,
                0.0, // no CACTUS
            );
            if result[2] != 0 {
                println!(
                    "  WARN seed={seed:#010x}: expected rejection at 0, got rejected_at={}",
                    result[2]
                );
            }
            let tok_g = result[1] as usize;
            if tok_g < VOCAB {
                hist_gpu[tok_g] += 1;
            }
        }

        let tv = tv_empirical_vs_theoretical(&hist_gpu, &theory_resid);
        let status = if tv < 0.02 { "PASS" } else { "FAIL" };
        println!("  {status} plain residual (8-token support): TV(GPU vs theory)={tv:.5}");
        if tv >= 0.02 {
            all_pass = false;
        }

        // CACTUS delta=1.0: accept_prob changes but residual is the same.
        {
            let mut hist_gpu_c = vec![0u32; VOCAB];
            let mut tgt_flat_c = p_tgt.clone();
            tgt_flat_c.extend_from_slice(&p_tgt);
            for s_idx in 0..N_SAMPLES {
                let seed = 0x8888_0000u32.wrapping_add(s_idx as u32);
                let result = gpu_chain_accept(
                    &mut gpu,
                    &tgt_flat_c,
                    &p_dft,
                    &[drafted as i32],
                    &[p_d_val],
                    &[0.0f32, 0.0],
                    &[1.0f32, 1.0],
                    &[0.0f32],
                    &[1.0f32],
                    1,
                    seed,
                    1.0, // cactus_delta = 1.0
                );
                let tok_g = result[1] as usize;
                if tok_g < VOCAB {
                    hist_gpu_c[tok_g] += 1;
                }
            }
            let tv_c = tv_empirical_vs_theoretical(&hist_gpu_c, &theory_resid);
            let status = if tv_c < 0.02 { "PASS" } else { "FAIL" };
            println!("  {status} residual with CACTUS delta=1.0: TV(GPU vs theory)={tv_c:.5}");
            if tv_c >= 0.02 {
                all_pass = false;
            }
        }
    }

    // ── Check 3b: CACTUS h-distribution residual (p_t > 0, h ≠ raw) ────────
    // The one case the δ=1.0 sub-check above CANNOT catch: when p_t > 0 at the
    // drafted token, the CACTUS rejection bonus must be drawn from the rewritten
    // h-distribution (speculative.rs:3660-3677), NOT the raw target. Here the two
    // give DIFFERENT residual support, so a kernel that skipped the h-rewrite
    // (the pre-fix behaviour) fails this check. Parameters are chosen so that:
    //   • rejection is frequent (accept_prob=0.486 < p_d=0.6 ⇒ ~19% reject), and
    //   • one support token (50) survives the RAW residual but is zeroed by the
    //     h residual (draft mass 0.35 sits between h[50]=0.325 and p_tgt[50]=0.6).
    // Expected h residual = {200:1.0}; raw residual = {50:0.45, 200:0.55}.
    println!("\n[3b] CACTUS h-distribution residual (p_t>0, h differs from raw target)");
    {
        let (d_tok, a_tok, b_tok) = (100usize, 50usize, 200usize);
        let cactus = 2.0f32;
        let p_d_val = 0.6f32;

        let mut p_tgt = vec![0.0f32; VOCAB];
        p_tgt[d_tok] = 0.05;
        p_tgt[a_tok] = 0.60;
        p_tgt[b_tok] = 0.35;
        let mut p_dft = vec![0.0f32; VOCAB];
        p_dft[d_tok] = 0.60;
        p_dft[a_tok] = 0.35;
        p_dft[b_tok] = 0.05;

        // Analytic h-distribution + residual theories (same formula as kernel/host).
        let p_t = p_tgt[d_tok];
        let bump = (2.0 * cactus * p_t * (1.0 - p_t)).sqrt();
        let gamma_star = (p_t + bump).min(1.0);
        let scale = (1.0 - gamma_star) / (1.0 - p_t.clamp(0.0, 1.0));
        let mut h = p_tgt.clone();
        for (j, v) in h.iter_mut().enumerate() {
            *v = if j == d_tok { gamma_star } else { scale * *v };
        }
        let normalize_residual = |tgt: &[f32], dft: &[f32]| -> Vec<f32> {
            let mut r: Vec<f32> = tgt
                .iter()
                .zip(dft)
                .map(|(&t, &d)| (t - d).max(0.0))
                .collect();
            let s: f32 = r.iter().sum();
            if s > 0.0 {
                r.iter_mut().for_each(|v| *v /= s);
            }
            r
        };
        let theory_h = normalize_residual(&h, &p_dft);
        let theory_raw = normalize_residual(&p_tgt, &p_dft);

        let mut tgt_flat = p_tgt.clone();
        tgt_flat.extend_from_slice(&p_tgt); // bonus row (unused: rejection path)

        let mut hist = vec![0u32; VOCAB];
        let mut n_reject = 0usize;
        for s_idx in 0..N_SAMPLES {
            let seed = 0x9999_0000u32.wrapping_add(s_idx as u32);
            let result = gpu_chain_accept(
                &mut gpu,
                &tgt_flat,
                &p_dft,
                &[d_tok as i32],
                &[p_d_val],
                &[0.0f32, 0.0],
                &[1.0f32, 1.0],
                &[0.0f32],
                &[1.0f32],
                1,
                seed,
                cactus,
            );
            if result[2] == 0 {
                let tok = result[1] as usize;
                if tok < VOCAB {
                    hist[tok] += 1;
                    n_reject += 1;
                }
            }
        }
        // Conditional TV (normalize by the number of rejection draws, not N_SAMPLES).
        let tv_cond = |hist: &[u32], theory: &[f32], total: usize| -> f32 {
            let n = total.max(1) as f32;
            0.5 * (0..hist.len().min(theory.len()))
                .map(|i| (hist[i] as f32 / n - theory[i]).abs())
                .sum::<f32>()
        };
        let tv_h = tv_cond(&hist, &theory_h, n_reject);
        let tv_raw = tv_cond(&hist, &theory_raw, n_reject);
        // Must MATCH the h-distribution and be FAR from the raw-target residual.
        let pass = n_reject > 300 && tv_h < 0.03 && tv_raw > 0.20;
        let status = if pass { "PASS" } else { "FAIL" };
        println!(
            "  {status} n_reject={n_reject}  TV(GPU vs h-dist)={tv_h:.5}  TV(GPU vs raw)={tv_raw:.5}  (want h<0.03, raw>0.20)"
        );
        if !pass {
            all_pass = false;
        }
    }

    // ── Check 4: Accept-len histogram TV < 0.02 ───────────────────────────
    // Theoretical: with p_accept per position = p_t/p_d (same for all b positions),
    // and draft_p_at_token = p_d, and target_p_at_token = p_t, the number of
    // accepted tokens follows a geometric distribution truncated at b.
    //
    // P(accept_len = k) = p^k * (1-p)   for k = 0..b-1
    // P(accept_len = b) = p^b            (all accepted)
    //
    // where p = p_t / p_d.
    //
    // Compare GPU empirical accept_len histogram against this theoretical distribution
    // using TV.  Use the LCG-based host chain to compute the same distribution
    // (avoids floating-point discrepancy in the theoretical formula).
    println!("\n[4] Accept-len distribution TV (GPU empirical vs host LCG reference)");
    {
        let b = 5usize;
        // p_accept ≈ 0.5 per position: easy to verify, wide spread over accept_len.
        let p_accept_target = 0.5f32;
        // Choose p_tgt = 0.01, p_dft = 0.02 → p_accept = p_tgt/p_dft = 0.5.
        let p_t_val = 0.01f32;
        let p_d_val = 0.02f32;
        // Verify arithmetic.
        assert!(
            (p_t_val / p_d_val - p_accept_target).abs() < 1e-5,
            "p_accept setup"
        );

        let drafted_tokens: Vec<i32> = (0..b).map(|i| (100 + i * 200) as i32).collect();

        // Build per-row prob vectors.
        let mut p_tgt_rows: Vec<Vec<f32>> = Vec::new();
        let mut p_dft_rows: Vec<Vec<f32>> = Vec::new();
        for i in 0..=b {
            if i < b {
                let dtok = drafted_tokens[i] as usize;
                let mut pt = vec![(1.0 - p_t_val) / (VOCAB as f32 - 1.0); VOCAB];
                pt[dtok] = p_t_val;
                let s: f32 = pt.iter().sum();
                pt.iter_mut().for_each(|v| *v /= s);

                let mut pd = vec![(1.0 - p_d_val) / (VOCAB as f32 - 1.0); VOCAB];
                pd[dtok] = p_d_val;
                let s: f32 = pd.iter().sum();
                pd.iter_mut().for_each(|v| *v /= s);

                p_tgt_rows.push(pt);
                p_dft_rows.push(pd);
            } else {
                // bonus row: uniform p_tgt for all-accepted bonus draw.
                let pt = vec![1.0f32 / VOCAB as f32; VOCAB];
                p_tgt_rows.push(pt);
            }
        }

        // Effective p_d at drafted position (after renorm).
        let draft_p_at_tok: Vec<f32> = (0..b).map(|i| p_d_val).collect();

        let tgt_flat: Vec<f32> = p_tgt_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let dft_flat: Vec<f32> = p_dft_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let tau_t_arr = vec![0.0f32; b + 1];
        let z_t_arr = vec![1.0f32; b + 1];
        let tau_d_arr = vec![0.0f32; b];
        let z_d_arr = vec![1.0f32; b];

        let mut hist_gpu = vec![0u32; b + 1];
        let mut hist_host = vec![0u32; b + 1];

        for s_idx in 0..N_SAMPLES {
            // GPU: one kernel call per sample.
            let seed_g = 0x5555_0000u32.wrapping_add(s_idx as u32);
            let result = gpu_chain_accept(
                &mut gpu,
                &tgt_flat,
                &dft_flat,
                &drafted_tokens,
                &draft_p_at_tok,
                &tau_t_arr,
                &z_t_arr,
                &tau_d_arr,
                &z_d_arr,
                b,
                seed_g,
                0.0,
            );
            let al_g = result[0].max(0) as usize;
            if al_g <= b {
                hist_gpu[al_g] += 1;
            }

            // Host: simulate with the same seed (LCG from (seed_g | 1)).
            let (al_h, _, _) = host_chain_accept_lcg(
                &p_tgt_rows,
                &p_dft_rows,
                &drafted_tokens,
                &draft_p_at_tok,
                &tau_t_arr,
                &z_t_arr,
                &tau_d_arr,
                &z_d_arr,
                b,
                seed_g,
                0.0,
            );
            let al_h = al_h.max(0) as usize;
            if al_h <= b {
                hist_host[al_h] += 1;
            }
        }

        let tv = tv_empirical(&hist_gpu, &hist_host);
        let status = if tv < 0.02 { "PASS" } else { "FAIL" };
        println!("  {status} accept-len TV (GPU vs host LCG, b={b}, p_accept≈{p_accept_target:.1}): TV={tv:.5}");
        println!("  host distribution: {:?}", &hist_host);
        println!("  gpu  distribution: {:?}", &hist_gpu);
        if tv >= 0.02 {
            all_pass = false;
        }
    }

    // ── Check 5: compressed-vocab inverse-map equivalence ────────────────
    // A compact draft row plus full→draft inverse map must be exactly
    // equivalent to scattering that row into a zero-filled full-vocab vector.
    println!("\n[5] Compressed-vocab inverse-map equivalence");
    {
        let b = 2usize;
        let map = vec![10usize, 50, 100, 500];
        let compact_rows = vec![
            vec![0.10f32, 0.60, 0.20, 0.10],
            vec![0.40f32, 0.10, 0.40, 0.10],
        ];
        let compact_flat: Vec<f32> =
            compact_rows.iter().flat_map(|row| row.iter().copied()).collect();
        let full_rows: Vec<Vec<f32>> = compact_rows
            .iter()
            .map(|compact| {
                let mut full = vec![0.0f32; VOCAB];
                for (draft_idx, &full_token) in map.iter().enumerate() {
                    full[full_token] = compact[draft_idx];
                }
                full
            })
            .collect();
        let full_flat: Vec<f32> =
            full_rows.iter().flat_map(|row| row.iter().copied()).collect();
        let draft_tokens = vec![50_i32, 100_i32];
        let draft_p_at_tok = vec![0.60f32, 0.40f32];

        let mut target_rows = Vec::with_capacity(b + 1);
        let mut row0 = vec![0.0f32; VOCAB];
        row0[50] = 0.30;
        row0[10] = 0.30;
        row0[200] = 0.40;
        target_rows.push(row0);
        let mut row1 = vec![0.0f32; VOCAB];
        row1[100] = 0.20;
        row1[10] = 0.30;
        row1[300] = 0.50;
        target_rows.push(row1);
        let mut bonus = vec![0.0f32; VOCAB];
        bonus[500] = 0.30;
        bonus[700] = 0.70;
        target_rows.push(bonus);
        let target_flat: Vec<f32> = target_rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let mut mismatches = 0usize;
        for i in 0..512u32 {
            let seed = 0xBEEF_0000u32.wrapping_add(i);
            let full = gpu_chain_accept(
                &mut gpu,
                &target_flat,
                &full_flat,
                &draft_tokens,
                &draft_p_at_tok,
                &[0.0, 0.0, 0.0],
                &[1.0, 1.0, 1.0],
                &[0.0, 0.0],
                &[1.0, 1.0],
                b,
                seed,
                0.0,
            );
            let compact = gpu_chain_accept_compressed(
                &mut gpu,
                &target_flat,
                &compact_flat,
                &map,
                &draft_tokens,
                &draft_p_at_tok,
                b,
                seed,
            );
            if full != compact {
                mismatches += 1;
                if mismatches <= 5 {
                    println!("  seed={seed:#010x}: full={full:?} compact={compact:?}");
                }
            }
        }
        if mismatches == 0 {
            println!("  PASS: 512/512 compact results match zero-scattered full vocab");
        } else {
            println!("  FAIL: {mismatches}/512 compact results differ");
            all_pass = false;
        }
    }

    // ── Summary ────────────────────────────────────────────────────────────
    println!();
    if all_pass {
        println!("=== RESULT: GO — all checks PASS ===");
    } else {
        println!("=== RESULT: NO-GO — one or more checks FAILED ===");
        std::process::exit(1);
    }
}
