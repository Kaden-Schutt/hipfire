// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! bench_gemv_lowbit — decode-shaped throughput for the TQ2G128 / BQ1G128 GEMVs.
//!
//! Decode is weight-bandwidth bound, so the figure of merit is GiB/s of weight
//! bytes streamed, not FLOPs. Reports that directly so a kernel change can be
//! compared against the MQ4 path (~156 GiB/s on gfx1151) rather than against an
//! abstract number.
//!
//! Protocol: one warmup launch (JIT + caches), then N timed launches with a
//! device_synchronize around the batch. Shapes default to the Bonsai-27B decode
//! shapes (K=5120 hidden, M=5120 square proj and M=17408 FFN up).
//!
//! Usage: bench_gemv_lowbit [ternary|binary] [reps]

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;
use std::time::Instant;

const TQ2_SRC: &str = include_str!("../../../kernels/src/gemv_tq2g128.hip");
const BQ1_SRC: &str = include_str!("../../../kernels/src/gemv_bq1g128.hip");

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let which = argv.get(1).cloned().unwrap_or_else(|| "ternary".into());
    let reps: usize = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);

    let (name, src, blk_bytes, group) = match which.as_str() {
        "binary" => ("gemv_bq1g128", BQ1_SRC, 18usize, 128usize),
        _ => ("gemv_tq2g128", TQ2_SRC, 34usize, 128usize),
    };

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "bench_gemv_lowbit: arch={} kernel={name} reps={reps}",
        gpu.arch
    );
    gpu.ensure_kernel_public(name, src, name).expect("jit");

    // Bonsai-27B decode shapes: square projection and the FFN up/gate.
    for (m, k) in [
        (48usize, 5120usize),
        (6144usize, 5120usize),
        (5120usize, 6144usize),
        (5120usize, 5120usize),
        (17408usize, 5120usize),
        (5120usize, 17408usize),
        (248320usize, 5120usize),
    ] {
        let groups = k / group;
        let total = m * groups * blk_bytes;
        // Deterministic filler; values do not affect bandwidth.
        let packed: Vec<u8> = (0..total).map(|i| (i * 37 % 251) as u8).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

        let d_a = gpu.upload_raw(&packed, &[total]).expect("upload A");
        let d_x = gpu.upload_f32(&x, &[k]).expect("upload x");
        let d_y = gpu.zeros(&[m], DType::F32).expect("zeros y");

        let mut kb = KernargBlob::new();
        kb.push_ptr(d_a.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_x.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_y.buf.as_ptr() as *const c_void);
        kb.push_i32(m as i32);
        kb.push_i32(k as i32);
        kb.pad_to(16);

        let mut launch = || {
            gpu.launch_kernel_blob(name, [m as u32, 1, 1], [32, 1, 1], 0, kb.as_mut_slice())
                .expect("launch");
        };

        launch(); // warmup: JIT settle + first-touch
        gpu.hip.device_synchronize().expect("sync");

        let t0 = Instant::now();
        for _ in 0..reps {
            launch();
        }
        gpu.hip.device_synchronize().expect("sync");
        let secs = t0.elapsed().as_secs_f64();

        let gib = total as f64 / (1024.0 * 1024.0 * 1024.0);
        let bw = gib * reps as f64 / secs;
        println!(
            "{name} M={m:<6} K={k:<5} {gib:6.3} GiB/launch  {:7.3} ms/launch  {bw:7.1} GiB/s",
            secs * 1000.0 / reps as f64
        );
    }
}
