// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — MQ4 prefill: INT8 MMQ vs F16 WMMA on the Qwen3.5-0.8B shapes.
//
// A microbenchmark of the bare instructions put the ceiling at ~2.2x:
//
//     f16   164.1 TFLOPS
//     fp8   373.4 TFLOPS
//     iu8   363.9 TFLOPS
//
// but MMQ was tried before and canned because it did not reproduce. A ceiling
// is not a kernel. MMQ additionally pays:
//   * quantizing activations to int8 (Q8_1) every call,
//   * the zp * sum(x) affine correction,
//   * whatever LDS/tiling shape the MMQ kernel actually uses,
// and any of those can eat a 2.2x instruction-rate advantage whole.
//
// Both entry points take the IDENTICAL argument list, so this is a true
// drop-in swap over the same weights and activations. The weights are MQ4
// (136 B per 256-weight group) either way — MMQ is a prefill-phase kernel over
// the same bytes, not a different format.
//
// Accuracy is reported alongside speed because MMQ quantizes the ACTIVATIONS,
// which the F16 path does not. If the previous attempt was canned for quality
// rather than speed, that shows up here.
//
// Run: cargo run --release -p rdna-compute --example mq4_prefill_mmq_vs_f16

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

/// MQ4G256 rows: (K/256) groups of `[f32 scale][f32 zero][128 B nibbles]`.
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
    eprintln!("=== MQ4 prefill: INT8 MMQ vs F16 WMMA, Qwen3.5-0.8B gate_up ===");
    eprintln!("  arch={}", gpu.arch);
    eprintln!("  identical args, identical MQ4 weights — only the kernel differs");
    eprintln!("  negative delta = MMQ faster");
    eprintln!();

    // Qwen3.5-0.8B: hidden 1024, gate and up are 3072 each.
    let (gate_m, up_m, k) = (3072usize, 3072usize, 1024usize);
    let mut rng = Rng(0x4D4D_5134_9E37_79B9);

    let a_gate_h = build_mq4(&mut rng, gate_m, k);
    let a_up_h = build_mq4(&mut rng, up_m, k);
    let a_gate = gpu.upload_raw(&a_gate_h, &[a_gate_h.len()]).expect("gate");
    let a_up = gpu.upload_raw(&a_up_h, &[a_up_h.len()]).expect("up");

    println!(
        "{:<8} {:>11} {:>11} {:>9} {:>12} {:>11}",
        "batch", "f16 ms", "mmq ms", "delta", "f16 TFLOPS", "rel err"
    );
    println!("{}", "-".repeat(68));

    for &batch in &[16usize, 32, 64, 128, 256, 512] {
        let x_h: Vec<f32> = (0..batch * k).map(|_| rng.unit()).collect();
        let x = gpu.upload_f32(&x_h, &[batch, k]).expect("x");
        let yg1 = gpu.alloc_tensor(&[batch, gate_m], DType::F32).expect("yg1");
        let yu1 = gpu.alloc_tensor(&[batch, up_m], DType::F32).expect("yu1");
        let yg2 = gpu.alloc_tensor(&[batch, gate_m], DType::F32).expect("yg2");
        let yu2 = gpu.alloc_tensor(&[batch, up_m], DType::F32).expect("yu2");

        let f16 = |g: &mut Gpu, yg: &_, yu: &_| {
            g.gemm_gate_up_hfq4g256(&a_gate, &a_up, &x, yg, yu, gate_m, up_m, k, batch)
        };
        let mmq = |g: &mut Gpu, yg: &_, yu: &_| {
            g.gemm_gate_up_hfq4g256_mmq(&a_gate, &a_up, &x, yg, yu, gate_m, up_m, k, batch)
        };

        for _ in 0..25 {
            f16(&mut gpu, &yg1, &yu1).expect("f16 warm");
            mmq(&mut gpu, &yg2, &yu2).expect("mmq warm");
        }
        gpu.hip.device_synchronize().expect("sync");

        // Accuracy: MMQ quantizes activations, F16 does not. Treat F16 as ref.
        let r1 = gpu.download_f32(&yg1).expect("dl f16");
        let r2 = gpu.download_f32(&yg2).expect("dl mmq");
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in r1.iter().zip(r2.iter()) {
            let d = *a as f64 - *b as f64;
            num += d * d;
            den += (*a as f64) * (*a as f64);
        }
        let rel = (num / den.max(1e-30)).sqrt();

        const REPS: usize = 7;
        const IT: usize = 10;
        let mut ma = Vec::with_capacity(REPS);
        let mut mb = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let mut ta = Vec::with_capacity(IT);
            let mut tb = Vec::with_capacity(IT);
            for _ in 0..IT {
                let s = Instant::now();
                for _ in 0..4 {
                    f16(&mut gpu, &yg1, &yu1).expect("f16");
                }
                gpu.hip.device_synchronize().expect("sync");
                ta.push(s.elapsed().as_secs_f64() * 1e3 / 4.0);

                let s = Instant::now();
                for _ in 0..4 {
                    mmq(&mut gpu, &yg2, &yu2).expect("mmq");
                }
                gpu.hip.device_synchronize().expect("sync");
                tb.push(s.elapsed().as_secs_f64() * 1e3 / 4.0);
            }
            ma.push(median(&mut ta));
            mb.push(median(&mut tb));
        }
        let a = median(&mut ma);
        let b = median(&mut mb);
        // gate + up, each batch x m x k MACs = 2 flops.
        let flops = 2.0 * (gate_m + up_m) as f64 * k as f64 * batch as f64;
        println!(
            "{batch:<8} {a:>11.4} {b:>11.4} {:>+8.2}% {:>12.1} {rel:>11.3e}",
            (b / a - 1.0) * 100.0,
            flops / (a * 1e-3) / 1e12
        );
    }

    println!("{}", "-".repeat(68));
    println!();
    println!("Instruction-rate ceiling from the WMMA probe was ~2.2x (-55%).");
    println!("Anything short of that is MMQ overhead: Q8_1 activation quantization,");
    println!("the zp*sum(x) correction, and the MMQ tiling shape.");
    println!("rel err is MMQ vs the F16 path — it is NOT zero by construction,");
    println!("because MMQ quantizes the activations and F16 does not.");
}
