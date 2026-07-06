// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Chunked FP32 gated_delta_net parity test.
//!
//! Verifies the chunked (parallel) FP32 GDN kernel
//! (`gated_delta_net_f32_chunked`) is numerically EQUAL to the sequential
//! `gated_delta_net_f32_batch_seq` kernel on identical inputs, over several
//! (HD-fixed-128, T, CS) configs INCLUDING a T-not-multiple-of-CS tail.
//!
//! Both kernels mutate the S state in place, so each gets its OWN clone of the
//! same S0. Inputs: deterministic LCG q/k/v/gate/beta + S0; q and k are
//! L2-NORMALIZED per (token, head) on host BEFORE upload (un-normalized k
//! diverges — the recurrence's S absmax blows up). gate is a log-decay:
//! gate = -|N(0,1)|*0.1 - 0.01 (matches the oracle gen_head_inputs).
//!
//! Gate (non-negotiable): max|out_chunk - out_seq| < 1e-4 AND
//! max|S_chunk - S_seq| < 1e-4. The f64 oracle hit 1.3e-15; f32 GPU is looser
//! (reduction order, expf) but must stay well under 1e-4.
//!
//! Build / run:
//!   cargo run --release --features deltanet \
//!     --example gdn_chunk_parity -p rdna-compute

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("gdn_chunk_parity requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::DType;
    const HD: usize = 128;
    const N_HEADS: usize = 3;
    const TOL: f32 = 1e-4;

    let mut gpu = init_gpu();

    // (T, CS) configs — last one is a T-not-multiple-of-CS tail (37 = 16+16+5).
    let configs: &[(usize, usize)] = &[(16, 16), (32, 16), (32, 32), (37, 16), (33, 8)];

    let mut overall_ok = true;
    println!(
        "{:>4} {:>4} {:>6} {:>12} {:>12}  pass",
        "T", "CS", "heads", "out_max", "state_max"
    );

    for &(t_tokens, cs) in configs {
        // ---- deterministic inputs (LCG), L2-normalized q & k per (t, h) ----
        let mut rng = Lcg::new(0xC0FFEE ^ ((t_tokens as u64) << 8) ^ cs as u64);
        let mut q = vec![0f32; t_tokens * N_HEADS * HD];
        let mut k = vec![0f32; t_tokens * N_HEADS * HD];
        let mut v = vec![0f32; t_tokens * N_HEADS * HD];
        for x in q.iter_mut() {
            *x = rng.normal() * 0.5;
        }
        for x in k.iter_mut() {
            *x = rng.normal() * 0.5;
        }
        for x in v.iter_mut() {
            *x = rng.normal() * 0.5;
        }
        // L2-normalize q and k per (token, head) vector of length HD.
        l2_normalize_rows(&mut q, t_tokens * N_HEADS, HD);
        l2_normalize_rows(&mut k, t_tokens * N_HEADS, HD);

        let mut gate = vec![0f32; t_tokens * N_HEADS];
        let mut beta = vec![0f32; t_tokens * N_HEADS];
        for g in gate.iter_mut() {
            *g = -rng.normal().abs() * 0.1 - 0.01; // log-decay < 0
        }
        for b in beta.iter_mut() {
            *b = rng.uniform(); // (0,1)
        }
        let s0: Vec<f32> = (0..N_HEADS * HD * HD).map(|_| rng.normal() * 0.1).collect();

        // ---- upload, two SEPARATE S clones (both kernels advance in place) ----
        let q_gpu = gpu.upload_f32(&q, &[t_tokens, N_HEADS * HD]).unwrap();
        let k_gpu = gpu.upload_f32(&k, &[t_tokens, N_HEADS * HD]).unwrap();
        let v_gpu = gpu.upload_f32(&v, &[t_tokens, N_HEADS * HD]).unwrap();
        let g_gpu = gpu.upload_f32(&gate, &[t_tokens, N_HEADS]).unwrap();
        let b_gpu = gpu.upload_f32(&beta, &[t_tokens, N_HEADS]).unwrap();

        let s_seq = gpu.upload_f32(&s0, &[N_HEADS * HD * HD]).unwrap();
        let s_chunk = gpu.upload_f32(&s0, &[N_HEADS * HD * HD]).unwrap();
        let out_seq = gpu.zeros(&[t_tokens, N_HEADS * HD], DType::F32).unwrap();
        let out_chunk = gpu.zeros(&[t_tokens, N_HEADS * HD], DType::F32).unwrap();

        // ---- sequential reference ----
        gpu.gated_delta_net_f32_batch_seq(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &s_seq, &out_seq, t_tokens, N_HEADS, HD,
        )
        .unwrap();

        // ---- chunked (host loop over chunks lives inside the wrapper) ----
        gpu.gated_delta_net_f32_chunked(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &s_chunk, &out_chunk, t_tokens, N_HEADS, HD, cs,
        )
        .unwrap();

        let out_seq_h = gpu.download_f32(&out_seq).unwrap();
        let out_chunk_h = gpu.download_f32(&out_chunk).unwrap();
        let s_seq_h = gpu.download_f32(&s_seq).unwrap();
        let s_chunk_h = gpu.download_f32(&s_chunk).unwrap();

        let out_max = max_abs_diff(&out_seq_h, &out_chunk_h);
        let s_max = max_abs_diff(&s_seq_h, &s_chunk_h);
        let ok = out_max < TOL && s_max < TOL;
        overall_ok &= ok;
        println!(
            "{:>4} {:>4} {:>6} {:>12.3e} {:>12.3e}  {}",
            t_tokens,
            cs,
            N_HEADS,
            out_max,
            s_max,
            if ok { "OK" } else { "FAIL" }
        );

        for tns in [
            q_gpu, k_gpu, v_gpu, g_gpu, b_gpu, s_seq, s_chunk, out_seq, out_chunk,
        ] {
            gpu.free_tensor(tns).unwrap();
        }
    }

    // ---- optional perf bench: chunked wrapper vs batch_seq at prefill shapes ----
    if std::env::var("HIPFIRE_GDN_BENCH").as_deref() == Ok("1") {
        bench(&mut gpu);
    }

    if overall_ok {
        println!("\nPARITY OK (chunked == batch_seq, all configs incl. tail, tol {TOL:.0e})");
    } else {
        eprintln!("\nPARITY FAIL");
        std::process::exit(1);
    }
}

/// Time the chunked wrapper vs the sequential batch_seq kernel at realistic
/// prefill shapes (HD=128 fixed, n_heads=16). Isolates the GDN kernel: H2D of
/// the fresh S clone is OUTSIDE the sync-bracketed timed region; warm median of
/// 13 (the cold first-use JIT is the max, dropped by the median).
#[cfg(feature = "deltanet")]
fn bench(gpu: &mut rdna_compute::Gpu) {
    use rdna_compute::DType;
    use std::time::Instant;
    const HD: usize = 128;
    const N_HEADS: usize = 16;
    const ITERS: usize = 13;
    // Small-n = the spec-decode VERIFY shape (DFlash block ~8-16, DDTree
    // candidates, MTP K~4-5) — single/few chunks, latency-critical. Large-n =
    // prefill. Both matter; verify is the better-justified target.
    let shapes: &[(usize, usize)] = &[
        (4, 4),
        (8, 8),
        (16, 16),
        (24, 16),
        (32, 16),
        (32, 32),
        (128, 16),
        (256, 16),
        (512, 16),
    ];

    println!("\n=== BENCH  HD={HD} n_heads={N_HEADS}  (warm median-of-{ITERS}, ms) ===");
    println!(
        "{:>5} {:>4} {:>12} {:>12} {:>9}",
        "T", "CS", "seq_ms", "chunk_ms", "speedup"
    );

    for &(t_tokens, cs) in shapes {
        let mut rng = Lcg::new(0xBEEF ^ ((t_tokens as u64) << 8) ^ cs as u64);
        let mut q = vec![0f32; t_tokens * N_HEADS * HD];
        let mut k = vec![0f32; t_tokens * N_HEADS * HD];
        let mut v = vec![0f32; t_tokens * N_HEADS * HD];
        for x in q.iter_mut() {
            *x = rng.normal() * 0.5;
        }
        for x in k.iter_mut() {
            *x = rng.normal() * 0.5;
        }
        for x in v.iter_mut() {
            *x = rng.normal() * 0.5;
        }
        l2_normalize_rows(&mut q, t_tokens * N_HEADS, HD);
        l2_normalize_rows(&mut k, t_tokens * N_HEADS, HD);
        let mut gate = vec![0f32; t_tokens * N_HEADS];
        let mut beta = vec![0f32; t_tokens * N_HEADS];
        for g in gate.iter_mut() {
            *g = -rng.normal().abs() * 0.1 - 0.01;
        }
        for b in beta.iter_mut() {
            *b = rng.uniform();
        }
        let s0: Vec<f32> = (0..N_HEADS * HD * HD).map(|_| rng.normal() * 0.1).collect();

        let q_gpu = gpu.upload_f32(&q, &[t_tokens, N_HEADS * HD]).unwrap();
        let k_gpu = gpu.upload_f32(&k, &[t_tokens, N_HEADS * HD]).unwrap();
        let v_gpu = gpu.upload_f32(&v, &[t_tokens, N_HEADS * HD]).unwrap();
        let g_gpu = gpu.upload_f32(&gate, &[t_tokens, N_HEADS]).unwrap();
        let b_gpu = gpu.upload_f32(&beta, &[t_tokens, N_HEADS]).unwrap();
        let out = gpu.zeros(&[t_tokens, N_HEADS * HD], DType::F32).unwrap();

        let mut seq_t = Vec::with_capacity(ITERS);
        let mut chunk_t = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let s = gpu.upload_f32(&s0, &[N_HEADS * HD * HD]).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            gpu.gated_delta_net_f32_batch_seq(
                &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &s, &out, t_tokens, N_HEADS, HD,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            seq_t.push(t0.elapsed().as_secs_f64() * 1e3);
            gpu.free_tensor(s).unwrap();
        }
        for _ in 0..ITERS {
            let s = gpu.upload_f32(&s0, &[N_HEADS * HD * HD]).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            gpu.gated_delta_net_f32_chunked(
                &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &s, &out, t_tokens, N_HEADS, HD, cs,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            chunk_t.push(t0.elapsed().as_secs_f64() * 1e3);
            gpu.free_tensor(s).unwrap();
        }
        seq_t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        chunk_t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let seq_ms = seq_t[ITERS / 2];
        let chunk_ms = chunk_t[ITERS / 2];
        println!(
            "{:>5} {:>4} {:>12.4} {:>12.4} {:>8.2}x",
            t_tokens,
            cs,
            seq_ms,
            chunk_ms,
            seq_ms / chunk_ms
        );

        for tns in [q_gpu, k_gpu, v_gpu, g_gpu, b_gpu, out] {
            gpu.free_tensor(tns).unwrap();
        }
    }
}

#[cfg(feature = "deltanet")]
fn init_gpu() -> rdna_compute::Gpu {
    rdna_compute::Gpu::init().expect("GPU init")
}

#[cfg(feature = "deltanet")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "deltanet")]
fn l2_normalize_rows(data: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

/// Tiny deterministic LCG + Box-Muller, so the harness needs no rand crate.
#[cfg(feature = "deltanet")]
struct Lcg {
    state: u64,
}

#[cfg(feature = "deltanet")]
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg {
            state: seed.wrapping_mul(0x2545F4914F6CDD1D) | 1,
        }
    }
    fn next_u32(&mut self) -> u32 {
        // LCG (Numerical Recipes) then take the high bits.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn uniform(&mut self) -> f32 {
        // (0,1) open-ish; +1 avoids exact 0.
        ((self.next_u32() as f64 + 1.0) / 4294967297.0) as f32
    }
    fn normal(&mut self) -> f32 {
        // Box-Muller using two uniforms.
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0f32 * u1.ln()).sqrt();
        r * (2.0f32 * std::f32::consts::PI * u2).cos()
    }
}
