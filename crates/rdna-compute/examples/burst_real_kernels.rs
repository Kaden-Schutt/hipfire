// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Time the REAL rdna-compute kernels in burst mode (no per-call sync,
//! no profiling, just back-to-back launches with one device sync at the
//! end). Compares against the bandwidth profiler's 9-16 µs/kernel
//! measurements which use per-kernel event_synchronize.
//!
//! The question: when production runs without per-kernel sync, are the
//! kernels actually 3 µs each (the dispatch floor) or 9 µs (matching the
//! profile)? The answer determines whether hipGraph can help at all.

use rdna_compute::Gpu;
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let dim = 4096usize;

    let init_a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let init_b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.0007 + 0.5).collect();
    let init_w: Vec<f32> = (0..dim).map(|i| 1.0 + ((i % 16) as f32) * 0.01).collect();
    let a = gpu.upload_f32(&init_a, &[dim]).unwrap();
    let b = gpu.upload_f32(&init_b, &[dim]).unwrap();
    let c = gpu.upload_f32(&init_a, &[dim]).unwrap();
    let weight = gpu.upload_f32(&init_w, &[dim]).unwrap();
    let scratch = gpu.upload_f32(&vec![0.0; dim], &[dim]).unwrap();

    let n_launches = 1000u32;
    let warmup = 200u32;

    eprintln!("=== Real rdna-compute kernels in burst mode (no sync between) ===\n");
    eprintln!("dim={dim}, n_launches={n_launches}, warmup={warmup}");

    macro_rules! bench {
        ($label:expr, $body:expr) => {{
            for _ in 0..warmup { $body; }
            gpu.hip.device_synchronize().unwrap();

            let t = Instant::now();
            for _ in 0..n_launches { $body; }
            gpu.hip.device_synchronize().unwrap();
            let total = t.elapsed().as_secs_f64() * 1_000_000.0;
            let per_call = total / n_launches as f64;
            eprintln!("[{:25}] {n_launches} launches in {total:7.1} µs → {per_call:5.2} µs/call", $label);
        }};
    }

    // ─── Single kernel burst (each kernel back-to-back) ─
    eprintln!("\n--- Single-kernel bursts ---");
    bench!("rmsnorm_f32", { gpu.rmsnorm_f32(&a, &weight, &scratch, 1e-6).unwrap(); });
    bench!("mul_f32", { gpu.mul_f32(&a, &b, &c).unwrap(); });
    bench!("add_inplace_f32", { gpu.add_inplace_f32(&a, &b).unwrap(); });
    bench!("silu_mul_f32", { gpu.silu_mul_f32(&a, &b, &scratch).unwrap(); });
    #[cfg(feature = "deltanet")]
    {
        bench!("sigmoid_f32", { gpu.sigmoid_f32(&scratch).unwrap(); });
        bench!("scale_f32", { gpu.scale_f32(&a, 0.5).unwrap(); });
    }

    // ─── Mixed dependent chain (mimics non-GEMV layer pattern) ─
    eprintln!("\n--- Mixed dependent chain (5 kernels per iteration) ---");
    bench!("mixed-5 dependent", {
        gpu.rmsnorm_f32(&a, &weight, &scratch, 1e-6).unwrap();
        #[cfg(feature = "deltanet")] { gpu.sigmoid_f32(&scratch).unwrap(); }
        gpu.mul_f32(&scratch, &b, &c).unwrap();
        gpu.add_inplace_f32(&c, &b).unwrap();
        gpu.silu_mul_f32(&a, &b, &scratch).unwrap();
    });

    eprintln!("\n--- Mixed dependent chain (10 kernels per iteration) ---");
    bench!("mixed-10 dependent", {
        gpu.rmsnorm_f32(&a, &weight, &scratch, 1e-6).unwrap();
        #[cfg(feature = "deltanet")] { gpu.sigmoid_f32(&scratch).unwrap(); }
        gpu.mul_f32(&scratch, &b, &c).unwrap();
        gpu.add_inplace_f32(&c, &b).unwrap();
        gpu.silu_mul_f32(&a, &b, &scratch).unwrap();
        gpu.rmsnorm_f32(&scratch, &weight, &c, 1e-6).unwrap();
        #[cfg(feature = "deltanet")] { gpu.sigmoid_f32(&c).unwrap(); }
        gpu.mul_f32(&c, &b, &scratch).unwrap();
        gpu.add_inplace_f32(&scratch, &a).unwrap();
        gpu.silu_mul_f32(&a, &scratch, &c).unwrap();
    });

    eprintln!("\nNote: divide by # of kernels per iteration to get per-kernel cost.");
    eprintln!("Compare to bandwidth-ceiling profiler (which adds event_sync per kernel):");
    eprintln!("  rmsnorm_f32: 16.42 µs/call");
    eprintln!("  silu_mul_f32:  9.15 µs/call");
    eprintln!("  sigmoid_f32:   9.04 µs/call");
    eprintln!("  mul_f32:       9.20 µs/call");
}
