// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity check for the gfx1103 wave-reduced (no heavy-LDS) reduction kernels:
//! rmsnorm, softmax, layernorm, max_prob.
//!
//! Each op runs through the live dispatcher and is compared against an f64 CPU
//! reference. On gfx1103 (default) this exercises the new `*_gfx1103` kernels;
//! with `HIPFIRE_FORCE_GENERIC=1` it exercises the generic LDS kernels, so the
//! same harness proves both paths agree with the reference.
//!
//!   cargo run --release -p hipfire-rdna --example parity_reductions_gfx1103
//!   HIPFIRE_FORCE_GENERIC=1 cargo run --release -p hipfire-rdna \
//!       --example parity_reductions_gfx1103

use hipfire_rdna::Gpu;

// Cheap deterministic pseudo-random in [-1, 1); avoids an rng dep.
fn pseudo(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn cpu_rmsnorm(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / n;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * n..r * n + n];
        let mut ss = 0.0f64;
        for &v in row {
            ss += (v as f64) * (v as f64);
        }
        let inv = 1.0f64 / ((ss / n as f64) + eps as f64).sqrt();
        for i in 0..n {
            out[r * n + i] = (row[i] as f64 * w[i] as f64 * inv) as f32;
        }
    }
    out
}

fn cpu_layernorm(x: &[f32], g: &[f32], b: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / n;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * n..r * n + n];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let inv = 1.0f64 / (var + eps as f64).sqrt();
        for i in 0..n {
            out[r * n + i] = (g[i] as f64 * (row[i] as f64 - mean) * inv + b[i] as f64) as f32;
        }
    }
    out
}

fn cpu_softmax(x: &[f32], n: usize) -> Vec<f32> {
    let rows = x.len() / n;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * n..r * n + n];
        let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let mut s = 0.0f64;
        let mut e = vec![0.0f64; n];
        for i in 0..n {
            e[i] = (row[i] as f64 - m).exp();
            s += e[i];
        }
        for i in 0..n {
            out[r * n + i] = (e[i] / s) as f32;
        }
    }
    out
}

fn cpu_max_prob(logits: &[f32]) -> f32 {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let s: f64 = logits.iter().map(|&v| (v as f64 - m).exp()).sum();
    (1.0f64 / s) as f32
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let forced = std::env::var("HIPFIRE_FORCE_GENERIC").is_ok();
    println!(
        "force_generic={forced} (path: {})",
        if forced {
            "generic LDS kernels"
        } else {
            "arch-selected kernels"
        }
    );

    let eps = 1e-6f32;
    let tol = 3e-4f32; // f32 accumulation slack; reduce order differs
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut worst = 0.0f32;
    let mut fails = 0;

    // (n, batch)
    let cases: &[(usize, usize)] = &[(128, 1), (256, 1), (2048, 1), (5120, 1), (3072, 5), (896, 11)];

    // ── rmsnorm ──────────────────────────────────────────────────────────
    for &(n, batch) in cases {
        let total = n * batch;
        let x: Vec<f32> = (0..total).map(|_| pseudo(&mut seed)).collect();
        let w: Vec<f32> = (0..n).map(|_| 0.5 + 0.5 * pseudo(&mut seed).abs()).collect();
        let refv = cpu_rmsnorm(&x, &w, n, eps);
        let xg = gpu.upload_f32(&x, &[batch, n]).unwrap();
        let wg = gpu.upload_f32(&w, &[n]).unwrap();
        let og = gpu.upload_f32(&vec![0.0; total], &[batch, n]).unwrap();
        if batch == 1 {
            gpu.rmsnorm_f32(&xg, &wg, &og, eps).unwrap();
        } else {
            gpu.rmsnorm_batched(&xg, &wg, &og, batch, n, eps).unwrap();
        }
        let err = max_abs_err(&gpu.download_f32(&og).unwrap(), &refv);
        worst = worst.max(err);
        let ok = err < tol;
        fails += !ok as i32;
        println!("  rmsnorm    n={n:5} batch={batch:3} err={err:.3e} {}", if ok { "ok" } else { "FAIL" });
    }

    // ── layernorm (batched API; batch=1 too) ─────────────────────────────
    for &(n, batch) in cases {
        let total = n * batch;
        let x: Vec<f32> = (0..total).map(|_| pseudo(&mut seed)).collect();
        let g: Vec<f32> = (0..n).map(|_| 0.5 + 0.5 * pseudo(&mut seed).abs()).collect();
        let b: Vec<f32> = (0..n).map(|_| pseudo(&mut seed)).collect();
        let refv = cpu_layernorm(&x, &g, &b, n, eps);
        let xg = gpu.upload_f32(&x, &[batch, n]).unwrap();
        let gg = gpu.upload_f32(&g, &[n]).unwrap();
        let bg = gpu.upload_f32(&b, &[n]).unwrap();
        let og = gpu.upload_f32(&vec![0.0; total], &[batch, n]).unwrap();
        gpu.layernorm_batched(&xg, &gg, &bg, &og, batch, n, eps).unwrap();
        let err = max_abs_err(&gpu.download_f32(&og).unwrap(), &refv);
        worst = worst.max(err);
        let ok = err < tol;
        fails += !ok as i32;
        println!("  layernorm  n={n:5} batch={batch:3} err={err:.3e} {}", if ok { "ok" } else { "FAIL" });
    }

    // ── softmax (in-place, per row) ──────────────────────────────────────
    for &(n, batch) in cases {
        let total = n * batch;
        // scale up so exp() spread is non-trivial
        let x: Vec<f32> = (0..total).map(|_| pseudo(&mut seed) * 8.0).collect();
        let refv = cpu_softmax(&x, n);
        let xg = gpu.upload_f32(&x, &[batch, n]).unwrap();
        gpu.softmax_f32(&xg).unwrap();
        let err = max_abs_err(&gpu.download_f32(&xg).unwrap(), &refv);
        worst = worst.max(err);
        let ok = err < tol;
        fails += !ok as i32;
        println!("  softmax    n={n:5} batch={batch:3} err={err:.3e} {}", if ok { "ok" } else { "FAIL" });
    }

    // ── max_prob (single block over vocab) ───────────────────────────────
    for &vocab in &[128usize, 2048, 32000, 152064] {
        let logits: Vec<f32> = (0..vocab).map(|_| pseudo(&mut seed) * 10.0).collect();
        let refv = cpu_max_prob(&logits);
        let lg = gpu.upload_f32(&logits, &[vocab]).unwrap();
        let rg = gpu.upload_f32(&[0.0], &[1]).unwrap();
        gpu.max_prob(&lg, &rg, vocab).unwrap();
        let got = gpu.download_f32(&rg).unwrap()[0];
        let err = (got - refv).abs();
        worst = worst.max(err);
        // max_prob is a probability in (0,1]; relative tol is what matters
        let ok = err < 3e-5 * refv.max(1e-6);
        fails += !ok as i32;
        println!("  max_prob   vocab={vocab:6}     ref={refv:.6} got={got:.6} err={err:.2e} {}", if ok { "ok" } else { "FAIL" });
    }

    // ── argmax (value+index; lowest index on ties) ──────────────────────
    fn cpu_argmax(row: &[f32]) -> u32 {
        let mut best = f32::MIN;
        let mut bi = 0u32;
        for (i, &v) in row.iter().enumerate() {
            if v > best {
                best = v;
                bi = i as u32;
            }
        }
        bi
    }
    for &n in &[128usize, 2048, 32000, 152064] {
        let data: Vec<f32> = (0..n).map(|_| pseudo(&mut seed) * 5.0).collect();
        let refv = cpu_argmax(&data);
        let dg = gpu.upload_f32(&data, &[n]).unwrap();
        let got = gpu.argmax_f32(&dg, n).unwrap();
        let ok = got == refv;
        fails += !ok as i32;
        println!("  argmax     n={n:6}     ref={refv} got={got} {}", if ok { "ok" } else { "FAIL" });
    }
    // crafted tie: two equal maxima; lowest index must win (matches generic)
    {
        let n = 4096usize;
        let mut data: Vec<f32> = (0..n).map(|_| pseudo(&mut seed).abs() * 0.5).collect();
        data[900] = 9.0;
        data[3100] = 9.0; // equal max; expect index 900
        let dg = gpu.upload_f32(&data, &[n]).unwrap();
        let got = gpu.argmax_f32(&dg, n).unwrap();
        let ok = got == 900;
        fails += !ok as i32;
        println!("  argmax-tie n={n:6}     ref=900 got={got} {}", if ok { "ok" } else { "FAIL" });
    }
    // batched argmax
    {
        let (batch, n) = (9usize, 3000usize);
        let mut data: Vec<f32> = (0..batch * n).map(|_| pseudo(&mut seed) * 5.0).collect();
        // ensure each row has a clean unique max
        let mut refs = vec![0u32; batch];
        for r in 0..batch {
            data[r * n + (r * 137 % n)] = 100.0 + r as f32;
            refs[r] = cpu_argmax(&data[r * n..r * n + n]);
        }
        let dg = gpu.upload_f32(&data, &[batch, n]).unwrap();
        let rg = gpu.upload_f32(&vec![0.0; batch], &[batch]).unwrap();
        gpu.argmax_f32_batched(&dg, &rg, n, batch).unwrap();
        // result is i32 stored in an f32-typed tensor buffer; read raw
        let raw = gpu.download_raw(&rg, batch * 4).unwrap();
        let got: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let ok = got == refs;
        fails += !ok as i32;
        println!("  argmax-bat batch={batch} n={n}   ref={refs:?} got={got:?} {}", if ok { "ok" } else { "FAIL" });
    }

    if fails == 0 {
        println!("OK — all reduction cases within tol (worst abs {worst:.3e})");
    } else {
        eprintln!("PARITY FAIL — {fails} case(s) out of tolerance");
        std::process::exit(1);
    }
}
