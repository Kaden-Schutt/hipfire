// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! F16-weight GEMM/GEMV cross-arch parity matrix (#26 ds4 EP divergence
//! hunt → task #41 gemm_f16_tiled fix).
//!
//! The ds4 compressor projections are F16 and go through
//! `GemvFamily::run_auto` (the same call `gemv_auto` makes in
//! hipfire-arch-deepseek4). On gfx1201 the compressor GEMV output came
//! back ~9x too large because the pre-WMMA fallback chain bottomed out
//! in `gemm_f16_tiled`, whose original inner loop multi-counted K
//! (overlapping tid*4-stride reads + a per-lane contiguous tail).
//!
//! This probe isolates the kernel stack from the model: deterministic
//! W[m,k] (F16) and X[n,k] (F32), CPU f32 reference, then three GPU
//! paths per shape:
//!   1. dispatch `run_auto` (what the compressor actually calls; n=1 only)
//!   2. direct `gemm_f16_batched_lmhead` (batch=n, writes Y[n,m])
//!   3. direct `gemm_f16_tiled` (writes Y[m,n]; arbitrary K incl. tails)
//!
//! The WMMA paths behind 1./2. require K % 32 == 0; cells outside a
//! path's contract are reported as `skip`. PASS = max_rel_err < 5e-2
//! for every executed cell (f16 weights; lmhead also rounds X to f16).
//!
//! max_rel_err denominator: max(|ref_i|, rms(ref)) — near-zero outputs
//! are judged against the typical output magnitude, so f16-X rounding
//! noise on a ~0 element doesn't masquerade as kernel error, while a
//! genuinely broken kernel (e.g. the old ~10x blowup) still reports
//! errors >> 1.
//!
//! Run: cargo run --release -p hipfire-runtime --example f16_gemv_parity
//! Bench section: append `--bench` for 200-iter us/call timings.

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let frac = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (frac | 0x80_0000) >> (1 - e + 13);
        return sign | m as u16;
    }
    sign | ((e as u16) << 10) | ((frac >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let b = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // subnormal
            let mut e = 127 - 15 - 10;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (((e + 10) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(b)
}

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

/// max_rel_err with denominator max(|ref_i|, rms(ref)); returns (max_rel, l2_ratio).
fn errs(y: &[f32], y_ref: &[f32]) -> (f64, f64) {
    let rms = (l2(y_ref) / (y_ref.len() as f64).sqrt()).max(1e-6);
    let mut max_rel = 0f64;
    for (a, b) in y.iter().zip(y_ref) {
        let d = (*a as f64 - *b as f64).abs() / (b.abs() as f64).max(rms);
        if d > max_rel {
            max_rel = d;
        }
    }
    (max_rel, l2(y) / l2(y_ref))
}

fn main() {
    use rdna_compute::{DType, Gpu};

    let bench = std::env::args().any(|a| a == "--bench");
    const PASS_THRESH: f64 = 5e-2;

    // (m, k, n) shape matrix. k=511 exercises the stride-32 tail path;
    // n>1 exercises the 2-D grid; shapes mirror real consumers
    // (ds4 compressor m=256/k=4096, vision encoder k=1152/4304).
    let shapes: &[(usize, usize, usize)] = &[
        (256, 4096, 1),
        (128, 4096, 1),
        (1024, 1536, 1),
        (256, 1152, 4),
        (512, 4304, 16),
        (64, 512, 1),
        (256, 511, 1),
        (256, 4096, 64),
    ];

    let mut gpu = Gpu::init().expect("gpu init");
    let wmma = gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12();
    println!(
        "arch={} has_wmma_w32={} has_wmma_w32_gfx12={}",
        gpu.arch,
        gpu.arch_caps.has_wmma_w32(),
        gpu.arch_caps.has_wmma_w32_gfx12()
    );
    println!(
        "{:>16} {:>28} {:>14} {:>10}  {}",
        "shape (m,k,n)", "path", "max_rel_err", "l2_ratio", "verdict"
    );

    let mut failures = 0usize;

    for &(m, k, n) in shapes {
        // Deterministic LCG fill, same on every box, re-seeded per shape.
        let mut s: u64 = 0x1234_5678_9abc_def0 ^ ((m as u64) << 40 | (k as u64) << 16 | n as u64);
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let w_f32: Vec<f32> = (0..m * k).map(|_| next() * 0.1).collect();
        let x_f32: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let w_f16_bits: Vec<u16> = w_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let w_bytes: Vec<u8> = w_f16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        // CPU reference: f32 accumulate over f16-rounded weights, [m][n].
        let mut y_ref_mn = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f32;
                for j in 0..k {
                    acc += f16_bits_to_f32(w_f16_bits[r * k + j]) * x_f32[c * k + j];
                }
                y_ref_mn[r * n + c] = acc;
            }
        }
        // Transposed reference [n][m] for the lmhead layout.
        let mut y_ref_nm = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                y_ref_nm[c * m + r] = y_ref_mn[r * n + c];
            }
        }

        let mut w_t = gpu.upload_raw(&w_bytes, &[m, k]).expect("upload w");
        w_t.dtype = DType::F16;
        let x_t = gpu.upload_f32(&x_f32, &[n, k]).expect("upload x");
        let y_t = gpu.upload_f32(&vec![0f32; m * n], &[m * n]).expect("upload y");

        let shape_str = format!("({m},{k},{n})");
        let mut report = |name: &str, y: &[f32], y_ref: &[f32]| {
            let (max_rel, ratio) = errs(y, y_ref);
            let ok = max_rel < PASS_THRESH;
            if !ok {
                failures += 1;
            }
            println!(
                "{:>16} {:>28} {:>14.3e} {:>10.4} {}",
                shape_str,
                name,
                max_rel,
                ratio,
                if ok { "ok" } else { "FAIL" }
            );
        };

        // 1. dispatch run_auto — the exact compressor call shape (GEMV, n=1).
        //    WMMA arches route to a K%32==0 kernel; skip outside contract.
        if n == 1 && (k % 32 == 0 || !wmma) {
            use hipfire_dispatch::context::DispatchCtx;
            use hipfire_dispatch::families::gemv::WeightRef;
            let gemv = hipfire_runtime::llama::gemv_family();
            let ctx = DispatchCtx::new(&gpu);
            let wr = WeightRef {
                buf: &w_t,
                dtype: w_t.dtype,
                m,
                k,
                row_stride: 0,
                rotation: None,
                awq_scale: None,
            };
            gemv.run_auto(&ctx, &mut gpu, &wr, &x_t, &y_t).expect("run_auto");
            let y = gpu.download_f32(&y_t).expect("dl");
            report("dispatch run_auto", &y[..m], &y_ref_nm[..m]);
        } else {
            println!(
                "{:>16} {:>28} {:>14} {:>10}  skip ({})",
                shape_str,
                "dispatch run_auto",
                "-",
                "-",
                if n != 1 { "n>1: GEMV only" } else { "K%32!=0: WMMA contract" }
            );
        }

        // 2. direct gemm_f16_batched_lmhead (batch=n, Y[n,m]).
        //    mw16/mb8 WMMA require K%32==0; non-WMMA fallback handles any K.
        if k % 32 == 0 || !wmma {
            let _ = gpu.hip.memset(&y_t.buf, 0, m * n * 4);
            gpu.gemm_f16_batched_lmhead(&w_t, &x_t, &y_t, m, k, n).expect("lmhead");
            let y = gpu.download_f32(&y_t).expect("dl");
            report("gemm_f16_batched_lmhead", &y, &y_ref_nm);
        } else {
            println!(
                "{:>16} {:>28} {:>14} {:>10}  skip (K%32!=0: WMMA contract)",
                shape_str, "gemm_f16_batched_lmhead", "-", "-"
            );
        }

        // 3. direct gemm_f16_tiled (Y[m,n]; the kernel under test —
        //    arbitrary K, no skip).
        {
            let _ = gpu.hip.memset(&y_t.buf, 0, m * n * 4);
            gpu.gemm_f16_tiled(&w_t, &x_t, &y_t, m, k, n).expect("tiled");
            let y = gpu.download_f32(&y_t).expect("dl");
            report("gemm_f16_tiled", &y, &y_ref_mn);
        }
    }

    println!(
        "\nparity: {}",
        if failures == 0 {
            "PASS (all executed cells max_rel_err < 5e-2)".to_string()
        } else {
            format!("FAIL ({failures} cell(s) >= 5e-2)")
        }
    );

    if bench {
        println!("\nbench (200 iters, device_synchronize-bounded):");
        for &(m, k, n) in &[(1024usize, 1536usize, 64usize), (256, 4096, 1)] {
            let w_bytes: Vec<u8> = (0..m * k * 2).map(|i| (i % 251) as u8).collect();
            let mut w_t = gpu.upload_raw(&w_bytes, &[m, k]).expect("upload w");
            w_t.dtype = DType::F16;
            let x_t = gpu.upload_f32(&vec![0.25f32; n * k], &[n, k]).expect("upload x");
            let y_t = gpu.upload_f32(&vec![0f32; m * n], &[m * n]).expect("upload y");

            let mut time = |name: &str, f: &mut dyn FnMut(&mut Gpu)| {
                for _ in 0..10 {
                    f(&mut gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                let t0 = std::time::Instant::now();
                for _ in 0..200 {
                    f(&mut gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
                println!("  ({m},{k},{n}) {name:>28}: {us:9.2} us/call");
            };

            time("gemm_f16_tiled", &mut |g: &mut Gpu| {
                g.gemm_f16_tiled(&w_t, &x_t, &y_t, m, k, n).expect("tiled");
            });
            time("gemm_f16_batched_lmhead", &mut |g: &mut Gpu| {
                g.gemm_f16_batched_lmhead(&w_t, &x_t, &y_t, m, k, n).expect("lmhead");
            });
        }
    }

    if failures > 0 {
        std::process::exit(1);
    }
}
