// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — MQ4 prefill: gfx12 iu8 WMMA vs F16 WMMA (down_proj / residual).
//
// This is the test that was missing. The earlier MMQ comparison
// (mq4_prefill_mmq_vs_f16) measured the gate_up MMQ family, which is dp4a
// (`sdot4`/`sudot4`) — a VALU instruction, not a matrix one. It lost badly,
// and I then generalised that to "matrix low-precision won't help either"
// using a tiling argument. That was a hand-wave.
//
// gemm_hfq4g256_residual_mmq is different: on RDNA4 it dispatches to
// kernels/src/gemm_hfq4g256_residual_mmq.gfx12.hip, a dedicated single-wave
// 16-row-tile port built on wmma_i32_16x16x16_iu8 — the MATRIX instruction
// that the WMMA probe clocked at ~364 TFLOPS against F16's ~164.
//
// So this measures the real question: on an actual MQ4 prefill GEMM, does the
// 2.2x matrix-instruction ceiling survive contact with a kernel?
//
// Both entry points take identical arguments over identical MQ4 weights
// (136 B per 256-weight group). MMQ additionally quantizes activations to
// Q8_1 internally, so relative error against the F16 path is reported — it is
// nonzero by construction and is part of the trade, not a bug.
//
// Run: cargo run --release -p rdna-compute --example mq4_prefill_iu8wmma_vs_f16

use rdna_compute::{DType, Gpu};
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / 8388608.0) - 1.0
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn build_mq4(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let row = groups * 136;
    let mut v = vec![0u8; m * row];
    for c in v.chunks_mut(8) {
        let b = rng.next_u64().to_le_bytes();
        let n = c.len();
        c.copy_from_slice(&b[..n]);
    }
    for r in 0..m {
        for g in 0..groups {
            let o = r * row + g * 136;
            v[o..o + 4].copy_from_slice(&0.012f32.to_le_bytes());
            v[o + 4..o + 8].copy_from_slice(&(-0.09f32).to_le_bytes());
        }
    }
    v
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== MQ4 prefill: iu8 WMMA (matrix) vs F16 WMMA — Qwen3.5-0.8B down_proj ===");
    eprintln!("  arch={}", gpu.arch);
    eprintln!("  iu8 path = gemm_hfq4g256_residual_mmq -> gfx12 wmma_i32_16x16x16_iu8");
    eprintln!("  NOT the dp4a MMQ family measured in mq4_prefill_mmq_vs_f16");
    eprintln!("  negative delta = iu8 WMMA faster");
    eprintln!();

    // Qwen3.5-0.8B down_proj: M = hidden = 1024, K = intermediate = 3072.
    let (m, k) = (1024usize, 3072usize);
    let mut rng = Rng(0x4955_3877_9E37_79B9);
    let a_h = build_mq4(&mut rng, m, k);
    let a = gpu.upload_raw(&a_h, &[a_h.len()]).expect("weights");

    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>11} {:>11}",
        "batch", "f16 ms", "iu8 ms", "fp8 ms", "iu8 vs f16", "fp8 vs f16"
    );
    println!("{}", "-".repeat(78));

    for &batch in &[16usize, 32, 64, 128, 256, 512] {
        let x_h: Vec<f32> = (0..batch * k).map(|_| rng.unit()).collect();
        let x = gpu.upload_f32(&x_h, &[batch, k]).expect("x");
        // `residual` ACCUMULATES into y (y += A*x), so an uninitialised
        // output makes the result garbage — the first run of this bench
        // reported rel err = NaN for exactly that reason.
        let zero = vec![0.0f32; batch * m];
        let y1 = gpu.upload_f32(&zero, &[batch, m]).expect("y1");
        let y2 = gpu.upload_f32(&zero, &[batch, m]).expect("y2");
        let y3 = gpu.upload_f32(&zero, &[batch, m]).expect("y3");

        for _ in 0..25 {
            gpu.gemm_hfq4g256_residual(&a, &x, &y1, m, k, batch).expect("f16 warm");
            gpu.gemm_hfq4g256_residual_mmq(&a, &x, &y2, m, k, batch).expect("iu8 warm");
            gpu.gemm_hfq4g256_residual_fp8mmq(&a, &x, &y3, m, k, batch).expect("fp8 warm");
        }
        gpu.hip.device_synchronize().expect("sync");

        // Re-zero so the accuracy check sees exactly one accumulation each.
        gpu.upload_f32(&zero, &[batch, m]).expect("rezero");
        let yc1 = gpu.upload_f32(&zero, &[batch, m]).expect("yc1");
        let yc2 = gpu.upload_f32(&zero, &[batch, m]).expect("yc2");
        gpu.gemm_hfq4g256_residual(&a, &x, &yc1, m, k, batch).expect("f16 acc");
        gpu.gemm_hfq4g256_residual_mmq(&a, &x, &yc2, m, k, batch).expect("iu8 acc");
        let yc3 = gpu.upload_f32(&zero, &[batch, m]).expect("yc3");
        gpu.gemm_hfq4g256_residual_fp8mmq(&a, &x, &yc3, m, k, batch).expect("fp8 acc");
        gpu.hip.device_synchronize().expect("sync");
        let r1 = gpu.download_f32(&yc1).expect("dl f16");
        let r2 = gpu.download_f32(&yc2).expect("dl iu8");
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (p, q) in r1.iter().zip(r2.iter()) {
            let d = *p as f64 - *q as f64;
            num += d * d;
            den += (*p as f64) * (*p as f64);
        }
        let rel = (num / den.max(1e-30)).sqrt();
        let r3 = gpu.download_f32(&yc3).expect("dl fp8");
        let mut n3 = 0.0f64;
        for (p, q) in r1.iter().zip(r3.iter()) {
            let d = *p as f64 - *q as f64;
            n3 += d * d;
        }
        let rel3 = (n3 / den.max(1e-30)).sqrt();

        const REPS: usize = 7;
        const IT: usize = 10;
        let mut ma = Vec::with_capacity(REPS);
        let mut mb = Vec::with_capacity(REPS);
        let mut mc = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let mut ta = Vec::with_capacity(IT);
            let mut tb = Vec::with_capacity(IT);
            let mut tc = Vec::with_capacity(IT);
            for _ in 0..IT {
                let s = Instant::now();
                for _ in 0..4 {
                    gpu.gemm_hfq4g256_residual(&a, &x, &y1, m, k, batch).expect("f16");
                }
                gpu.hip.device_synchronize().expect("sync");
                ta.push(s.elapsed().as_secs_f64() * 1e3 / 4.0);

                let s = Instant::now();
                for _ in 0..4 {
                    gpu.gemm_hfq4g256_residual_mmq(&a, &x, &y2, m, k, batch).expect("iu8");
                }
                gpu.hip.device_synchronize().expect("sync");
                tb.push(s.elapsed().as_secs_f64() * 1e3 / 4.0);

                let s = Instant::now();
                for _ in 0..4 {
                    gpu.gemm_hfq4g256_residual_fp8mmq(&a, &x, &y3, m, k, batch).expect("fp8");
                }
                gpu.hip.device_synchronize().expect("sync");
                tc.push(s.elapsed().as_secs_f64() * 1e3 / 4.0);
            }
            ma.push(median(&mut ta));
            mb.push(median(&mut tb));
            mc.push(median(&mut tc));
        }
        let ta = median(&mut ma);
        let tb = median(&mut mb);
        let tc = median(&mut mc);
        let flops = 2.0 * m as f64 * k as f64 * batch as f64;
        println!(
            "{batch:<8} {ta:>10.4} {tb:>10.4} {tc:>10.4} {:>+10.2}% {:>10.2}%",
            (tb / ta - 1.0) * 100.0,
            (tc / ta - 1.0) * 100.0
        );
        println!(
            "{:<8} {:>10} {:>10.3e} {:>10.3e}  (rel err vs f16; f16 {:.1} TFLOPS)",
            "", "", rel, rel3, flops / (ta * 1e-3) / 1e12
        );
    }

    println!("{}", "-".repeat(68));
    println!();
    println!("Instruction ceiling from the WMMA probe: iu8 ~364 TF vs f16 ~164 TF (-55%).");
    println!("If f16 here sits far below 164 TF, these shapes are not MAC-bound and the");
    println!("ceiling cannot be collected regardless of which matrix instruction is used.");
}
