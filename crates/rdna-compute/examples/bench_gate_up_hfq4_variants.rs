// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Step-1 variant bench: HFQ4-G256 gate_up GEMM kernels head-to-head on the
//! prefill shape, timed + correctness vs the dot2 reference, %-of-WMMA-peak
//! (gfx1201 fp16 WMMA peak = 170.6 TFLOPS, microbenched 2026-06-08).
//! Run: cargo run --release --features deltanet -p rdna-compute --example bench_gate_up_hfq4_variants
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn build_hfq4g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let bpr = gpr * 136;
    let mut out = vec![0u8; m * bpr];
    let mix = |x: u64| {
        let h = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;
    for row in 0..m {
        for g in 0..gpr {
            let off = row * bpr + g * 136;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.01 + (((r1 as u32) % 4001) as f32) * 1e-5;
            let zero = (((r2 as u32) % 1500) as f32) * 1e-4 - 0.075;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte_i in 0..128 {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (byte_i as u64));
                out[off + 8 + byte_i] = (r & 0xff) as u8;
            }
        }
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let arch = gpu.arch.clone();
    let peak = 170.6_f64; // gfx1201 fp16 WMMA peak TFLOPS (microbenched)
    eprintln!("arch={arch}  peak(fp16 WMMA)={peak} TFLOPS");

    let gate_m = 16384usize;
    let up_m = 16384usize;
    let k = 5120usize;

    let a_gate = gpu.upload_raw(&build_hfq4g256(gate_m, k, 0xD4), &[gate_m, k]).unwrap();
    let a_up = gpu.upload_raw(&build_hfq4g256(up_m, k, 0xE5), &[up_m, k]).unwrap();

    for &n in &[64usize, 256, 512] {
        let total_m = gate_m + up_m;
        let flop = 2.0 * n as f64 * k as f64 * total_m as f64;
        let x_f32: Vec<f32> = (0..(n * k))
            .map(|i| { let b = (i / k) as i32; let kk = (i % k) as i32; ((b * 7 + kk * 11) % 31 - 15) as f32 * 0.05 })
            .collect();
        let x = gpu.upload_f32(&x_f32, &[n, k]).unwrap();
        let y_g = gpu.alloc_tensor(&[n, gate_m], DType::F32).unwrap();
        let y_u = gpu.alloc_tensor(&[n, up_m], DType::F32).unwrap();

        // reference
        gpu.gemm_gate_up_hfq4g256_dot2(&a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n).unwrap();
        let ref_g = gpu.download_f32(&y_g).unwrap();

        eprintln!("\n=== N={n}  (gate=up={gate_m} K={k})  FLOP={flop:.2e} ===");
        eprintln!("{:<14} {:>10} {:>9} {:>7} {:>10}", "variant", "us/call", "TFLOPS", "%peak", "max_rel");

        macro_rules! bench {
            ($label:expr, $m:ident) => {{
                let ok = gpu.$m(&a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n).is_ok();
                if !ok { eprintln!("{:<14} (call failed/skipped)", $label); }
                else {
                    for _ in 0..3 { let _ = gpu.$m(&a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n); }
                    gpu.hip.device_synchronize().unwrap();
                    let runs = 30;
                    let t0 = Instant::now();
                    for _ in 0..runs { let _ = gpu.$m(&a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n); }
                    gpu.hip.device_synchronize().unwrap();
                    let us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;
                    let tflops = flop / (us * 1e-6) / 1e12;
                    let cg = gpu.download_f32(&y_g).unwrap();
                    let mut mr = 0f32;
                    for (a, b) in cg.iter().zip(ref_g.iter()) { let r = (a - b).abs() / b.abs().max(1e-3); if r > mr { mr = r; } }
                    eprintln!("{:<14} {:>10.1} {:>9.1} {:>6.1}% {:>10.2e}", $label, us, tflops, tflops / peak * 100.0, mr);
                }
            }};
        }

        bench!("dot2(ref)", gemm_gate_up_hfq4g256_dot2);
        bench!("wmma_gfx12", gemm_gate_up_hfq4g256_wmma_gfx12);
        bench!("fp16", gemm_gate_up_hfq4g256_fp16);
        bench!("mmq", gemm_gate_up_hfq4g256_mmq);
        bench!("mmq_x16", gemm_gate_up_hfq4g256_mmq_x16);
        bench!("mmq_x32", gemm_gate_up_hfq4g256_mmq_x32);

        gpu.free_tensor(x).ok();
        gpu.free_tensor(y_g).ok();
        gpu.free_tensor(y_u).ok();
    }
}
