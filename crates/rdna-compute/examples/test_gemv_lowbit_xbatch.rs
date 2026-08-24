// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device parity for the x-batched TQ2-G128 / BQ1-G128 GEMVs.
//!
//! These are the prefill / multi-stream kernels: `y[b][row] = A[row] . x[b]`
//! for b in 0..B, reading each weight row ONCE rather than once per b.
//!
//! Checked against an independent CPU oracle (not against the scalar kernel),
//! so a shared misunderstanding of the block layout cannot cancel out.
//!
//! Shapes exercise both the 4-group unrolled body and the tail:
//!   - K=512 (4 groups)  -> quads=1, tail=0
//!   - K=384 (3 groups)  -> quads=0, tail=3   (all tail)
//!   - K=640 (5 groups)  -> quads=1, tail=1
//! and every batch size the kernel admits, B=1..=4.
//!
//! NEGATIVE CONTROL: the most plausible bug in an x-batched kernel is
//! ignoring `b` and dotting every row against x[0]. So after the parity pass
//! we re-check with a perturbed x[1..] and require the b>0 outputs to MOVE.
//! A kernel that reused x[0] would sail through parity when all x rows happen
//! to be equal, and this control is what makes that impossible.

use rdna_compute::{DType, Gpu};

const GROUP: usize = 128;
const TQ2_BLOCK: usize = 34;
const BQ1_BLOCK: usize = 18;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }
    fn f01(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn signed(&mut self) -> f32 {
        self.f01() * 2.0 - 1.0
    }
}

/// Truncating f32->f16 for positive normals. The oracle reads back through
/// `f16_to_f32` so both sides use the exact scale the kernel loads.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3ff) as u16;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | mant
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// Pack `m x k` weights. Returns (packed, per-element decoded weight).
/// `ternary` selects the 34B/3-level block or the 18B/2-level block.
fn pack(m: usize, k: usize, ternary: bool, rng: &mut Rng) -> (Vec<u8>, Vec<f32>) {
    let groups = k / GROUP;
    let blk = if ternary { TQ2_BLOCK } else { BQ1_BLOCK };
    let row_bytes = groups * blk;
    let mut packed = vec![0u8; m * row_bytes];
    let mut w = vec![0f32; m * k];

    for row in 0..m {
        for g in 0..groups {
            let d = 0.02 + rng.f01() * (1.5 - 0.02);
            let dh = f32_to_f16_bits(d);
            let d_rt = f16_to_f32(dh);
            let off = row * row_bytes + g * blk;
            packed[off] = (dh & 0xff) as u8;
            packed[off + 1] = (dh >> 8) as u8;

            for j in 0..GROUP {
                let idx = row * k + g * GROUP + j;
                if ternary {
                    let code = (rng.next_u32() % 3) as u8; // 0,1,2 -> -d,0,+d
                    packed[off + 2 + j / 4] |= code << ((j % 4) * 2);
                    w[idx] = (code as i32 - 1) as f32 * d_rt;
                } else {
                    let bit = (rng.next_u32() & 1) as u8; // 1 -> +d, 0 -> -d
                    packed[off + 2 + j / 8] |= bit << (j % 8);
                    w[idx] = if bit == 1 { d_rt } else { -d_rt };
                }
            }
        }
    }
    (packed, w)
}

#[allow(clippy::too_many_arguments)]
fn run_shape(gpu: &mut Gpu, ternary: bool, m: usize, k: usize, b: usize, seed: u64) -> bool {
    let name = if ternary { "tq2g128" } else { "bq1g128" };
    let mut rng = Rng(seed);
    let (packed, w) = pack(m, k, ternary, &mut rng);
    let xs: Vec<f32> = (0..b * k).map(|_| rng.signed()).collect();

    // CPU oracle: y[bi][row] = sum_j w[row][j] * x[bi][j]
    let mut expect = vec![0f32; b * m];
    for bi in 0..b {
        for row in 0..m {
            let mut acc = 0f32;
            for j in 0..k {
                acc += w[row * k + j] * xs[bi * k + j];
            }
            expect[bi * m + row] = acc;
        }
    }

    let d_a = gpu.upload_raw(&packed, &[packed.len()]).expect("upload A");
    let d_x = gpu.upload_f32(&xs, &[b * k]).expect("upload x");
    let d_y = gpu.zeros(&[b * m], DType::F32).expect("zeros y");

    let res = if ternary {
        gpu.gemv_tq2g128_xbatch(&d_a, &d_x, &d_y, m, k, b)
    } else {
        gpu.gemv_bq1g128_xbatch(&d_a, &d_x, &d_y, m, k, b)
    };
    res.expect("launch xbatch");
    let got = gpu.download_f32(&d_y).expect("download y");

    let max_abs = expect.iter().fold(0f32, |a, v| a.max(v.abs()));
    let tol = 1e-3 * (1.0 + max_abs);
    let mut worst = 0f32;
    for i in 0..b * m {
        worst = worst.max((got[i] - expect[i]).abs());
    }
    if worst >= tol {
        eprintln!("FAIL: {name} xbatch M={m} K={k} B={b} max_err={worst:e} >= tol={tol:e}");
        return false;
    }
    println!("PARITY OK {name} xbatch M={m:<4} K={k:<4} B={b} max_err={worst:e} tol={tol:e}");

    // NEG-CONTROL: perturb x[1..]; every b>0 output must move. Catches a
    // kernel that ignores b and reuses x[0].
    if b > 1 {
        let mut xs2 = xs.clone();
        for v in xs2.iter_mut().skip(k) {
            *v += 0.5;
        }
        let d_x2 = gpu.upload_f32(&xs2, &[b * k]).expect("upload x2");
        let d_y2 = gpu.zeros(&[b * m], DType::F32).expect("zeros y2");
        let res2 = if ternary {
            gpu.gemv_tq2g128_xbatch(&d_a, &d_x2, &d_y2, m, k, b)
        } else {
            gpu.gemv_bq1g128_xbatch(&d_a, &d_x2, &d_y2, m, k, b)
        };
        res2.expect("launch xbatch neg");
        let got2 = gpu.download_f32(&d_y2).expect("download y2");

        // b == 0 rows must be untouched...
        for row in 0..m {
            if (got2[row] - got[row]).abs() > tol {
                eprintln!("FAIL: {name} xbatch B={b} row {row}: b=0 output moved when only x[1..] changed");
                return false;
            }
        }
        // ...and every b > 0 row must have moved somewhere.
        let mut moved = 0usize;
        for bi in 1..b {
            for row in 0..m {
                if (got2[bi * m + row] - got[bi * m + row]).abs() > tol {
                    moved += 1;
                }
            }
        }
        if moved == 0 {
            eprintln!("FAIL: {name} xbatch B={b} perturbing x[1..] changed NOTHING — kernel is ignoring b");
            return false;
        }
        println!(
            "  NEG-CONTROL OK {name} B={b} ({moved}/{} b>0 outputs moved)",
            (b - 1) * m
        );
    }
    true
}

/// Same oracle, but for the tiled prefill GEMM. `n` deliberately includes
/// values that are NOT multiples of TILE_N so the partial-tile path is covered.
fn run_prefill(gpu: &mut Gpu, ternary: bool, m: usize, k: usize, n: usize, seed: u64) -> bool {
    let name = if ternary { "tq2g128" } else { "bq1g128" };
    let mut rng = Rng(seed);
    let (packed, w) = pack(m, k, ternary, &mut rng);
    let xs: Vec<f32> = (0..n * k).map(|_| rng.signed()).collect();

    let mut expect = vec![0f32; n * m];
    for ni in 0..n {
        for row in 0..m {
            let mut acc = 0f32;
            for j in 0..k {
                acc += w[row * k + j] * xs[ni * k + j];
            }
            expect[ni * m + row] = acc;
        }
    }

    let d_a = gpu.upload_raw(&packed, &[packed.len()]).expect("upload A");
    let d_x = gpu.upload_f32(&xs, &[n * k]).expect("upload x");
    let d_y = gpu.zeros(&[n * m], DType::F32).expect("zeros y");
    let res = if ternary {
        gpu.gemm_tq2g128_prefill(&d_a, &d_x, &d_y, m, k, n)
    } else {
        gpu.gemm_bq1g128_prefill(&d_a, &d_x, &d_y, m, k, n)
    };
    res.expect("launch prefill");
    let got = gpu.download_f32(&d_y).expect("download y");

    let max_abs = expect.iter().fold(0f32, |a, v| a.max(v.abs()));
    let tol = 1e-3 * (1.0 + max_abs);
    let mut worst = 0f32;
    for i in 0..n * m {
        worst = worst.max((got[i] - expect[i]).abs());
    }
    if worst >= tol {
        eprintln!("FAIL: {name} prefill M={m} K={k} N={n} max_err={worst:e} >= tol={tol:e}");
        return false;
    }
    println!("PARITY OK {name} prefill M={m:<4} K={k:<4} N={n:<4} max_err={worst:e} tol={tol:e}");
    true
}

/// WMMA prefill GEMM: same oracle, but activations are staged to F16 first,
/// so the tolerance has to admit F16 rounding of x (~1e-3 relative) rather
/// than the F32 paths' ~1e-6.
fn run_wmma(gpu: &mut Gpu, ternary: bool, m: usize, k: usize, n: usize, seed: u64) -> bool {
    let name = if ternary { "tq2g128" } else { "bq1g128" };
    let mut rng = Rng(seed);
    let (packed, w) = pack(m, k, ternary, &mut rng);
    let xs: Vec<f32> = (0..n * k).map(|_| rng.signed()).collect();
    // Round x through F16 so the oracle sees exactly what the kernel reads.
    let xs_rt: Vec<f32> = xs.iter().map(|v| f16_to_f32(f32_to_f16_bits(*v))).collect();

    let mut expect = vec![0f32; n * m];
    for ni in 0..n {
        for row in 0..m {
            let mut acc = 0f32;
            for j in 0..k {
                acc += w[row * k + j] * xs_rt[ni * k + j];
            }
            expect[ni * m + row] = acc;
        }
    }

    let d_a = gpu.upload_raw(&packed, &[packed.len()]).expect("upload A");
    // The kernel takes F32 and converts to F16 fragments in-register, so feed
    // it the F16-rounded values the oracle used -- that isolates the kernel's
    // arithmetic from the rounding, which the oracle already accounts for.
    let x_h = gpu.upload_f32(&xs_rt, &[n * k]).expect("upload x f32");
    let d_y = gpu.zeros(&[n * m], DType::F32).expect("zeros y");
    let res = if ternary {
        gpu.gemm_tq2g128_wmma(&d_a, &x_h, &d_y, m, k, n)
    } else {
        gpu.gemm_bq1g128_wmma(&d_a, &x_h, &d_y, m, k, n)
    };
    res.expect("launch wmma");
    let got = gpu.download_f32(&d_y).expect("download y");

    let max_abs = expect.iter().fold(0f32, |a, v| a.max(v.abs()));
    // A WMMA fragment accumulates 16 F16 products per step in F32; the A side
    // is exact (ternary/binary * an F16 scale), so the error is dominated by
    // F16 x rounding accumulated over K.
    let tol = 5e-3 * (1.0 + max_abs);
    let mut worst = 0f32;
    for i in 0..n * m {
        worst = worst.max((got[i] - expect[i]).abs());
    }
    if worst >= tol {
        eprintln!("FAIL: {name} wmma M={m} K={k} N={n} max_err={worst:e} >= tol={tol:e}");
        return false;
    }
    println!("PARITY OK {name} wmma    M={m:<4} K={k:<4} N={n:<4} max_err={worst:e} tol={tol:e}");
    true
}

/// Fused 4-way qkvza GEMM must equal four separate plain GEMMs on the same
/// activations. The plain path is itself oracle-verified above, so this is a
/// genuine equivalence check on the fusion/routing, which is where a fused
/// kernel actually goes wrong (rows routed to the wrong matrix or stride).
fn run_fused_qkvza(gpu: &mut Gpu, m: [usize; 4], k: usize, n: usize, seed: u64) -> bool {
    let mut rng = Rng(seed);
    let mut packed = Vec::new();
    let mut dev = Vec::new();
    for mi in m.iter() {
        let (p, _) = pack(*mi, k, true, &mut rng);
        let d = gpu.upload_raw(&p, &[p.len()]).expect("upload A");
        packed.push(p);
        dev.push(d);
    }
    let xs: Vec<f32> = (0..n * k).map(|_| rng.signed()).collect();
    let d_x = gpu.upload_f32(&xs, &[n * k]).expect("upload x");

    // Reference: four independent plain GEMMs.
    let mut refs = Vec::new();
    for (i, mi) in m.iter().enumerate() {
        let y = gpu.zeros(&[n * mi], DType::F32).expect("zeros");
        gpu.gemm_tq2g128_prefill(&dev[i], &d_x, &y, *mi, k, n)
            .expect("plain");
        refs.push(gpu.download_f32(&y).expect("dl"));
    }

    // Fused: one launch.
    let ys: Vec<_> = m
        .iter()
        .map(|mi| gpu.zeros(&[n * mi], DType::F32).expect("zeros"))
        .collect();
    gpu.gemm_qkvza_tq2g128(
        &dev[0], &dev[1], &dev[2], &dev[3], &d_x, &ys[0], &ys[1], &ys[2], &ys[3], m[0], m[1], m[2],
        m[3], k, n,
    )
    .expect("fused");

    for (i, mi) in m.iter().enumerate() {
        let got = gpu.download_f32(&ys[i]).expect("dl");
        let r = &refs[i];
        let max_abs = r.iter().fold(0f32, |a, v| a.max(v.abs()));
        let tol = 1e-3 * (1.0 + max_abs);
        let mut worst = 0f32;
        for j in 0..n * mi {
            worst = worst.max((got[j] - r[j]).abs());
        }
        if worst >= tol {
            eprintln!(
                "FAIL: fused qkvza matrix {i} (m={mi}) K={k} N={n} max_err={worst:e} >= tol={tol:e}"
            );
            return false;
        }
    }
    println!("PARITY OK fused qkvza  m={m:?} K={k:<4} N={n:<4} (== 4 separate GEMMs)");
    true
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch: {}", gpu.arch);
    let mut ok = true;
    let mut seed = 0xC0FFEE_u64;
    for &ternary in &[true, false] {
        // K=512 -> quads=1 tail=0; K=384 -> quads=0 tail=3; K=640 -> quads=1 tail=1
        for &(m, k) in &[(32usize, 512usize), (24, 384), (40, 640)] {
            for b in 1..=4usize {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ok &= run_shape(&mut gpu, ternary, m, k, b, seed | 1);
            }
        }
    }
    // Tiled prefill GEMM. N values straddle TILE_N=32: 1 (single token),
    // 31 (one partial tile), 32 (exact), 33 and 70 (full + partial).
    for &ternary in &[true, false] {
        for &(m, k) in &[(32usize, 512usize), (24, 384)] {
            for &n in &[1usize, 31, 32, 33, 70] {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ok &= run_prefill(&mut gpu, ternary, m, k, n, seed | 1);
            }
        }
    }

    // WMMA prefill. N straddles the 16-wide batch tile; M=24 straddles the
    // 16-wide row tile, so the edge-clamp paths are covered.
    for &ternary in &[true, false] {
        for &(m, k) in &[(32usize, 512usize), (24, 384)] {
            for &n in &[1usize, 15, 16, 17, 70] {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ok &= run_wmma(&mut gpu, ternary, m, k, n, seed | 1);
            }
        }
    }

    // Fused qkvza. Uneven m values on purpose: beta/alpha are tiny in the
    // real LA layer, and the routing arithmetic is where fusion breaks.
    for &n in &[1usize, 8, 33, 70] {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ok &= run_fused_qkvza(&mut gpu, [64, 32, 8, 8], 512, n, seed | 1);
    }

    if !ok {
        eprintln!("\nFAIL: x-batch parity/neg-control did not pass.");
        std::process::exit(1);
    }
    println!("ALL PASS");
}
