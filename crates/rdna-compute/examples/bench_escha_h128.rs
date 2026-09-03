// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throughput benchmark for `escha_h128_in` / `escha_h128_out`, kernel-only
//! (no host round trip): upload fixed input buffers once, then launch the
//! device-resident `Gpu::escha_h128` in a loop, syncing once at the end.
//!
//! These kernels run on every token, on both sides of every escha matmul —
//! unlike `escha_decode_tiles` (once per expert at load time), this is the
//! hot decode path. Used for the Task 8 naive-vs-parallel-butterfly
//! before/after measurement. Not wired into any gate.
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    // A production gate_up-sized activation: 2048 input channels (16 blocks
    // of 128), matching the packed_gu_e0_k2 golden fixture's `ic`.
    let n = 2048usize;
    let x: Vec<f32> = (0..n).map(|i| ((i * 37) as f32 * 0.017).sin()).collect();
    let rin: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { -0.0023 } else { 0.0023 })
        .collect();
    let mut rout: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
    rout[7] = 0.0;
    rout[1000] = 0.0;

    let mut gpu = Gpu::init().expect("gpu");

    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let rin_bytes: Vec<u8> = rin.iter().flat_map(|v| v.to_le_bytes()).collect();
    let rout_bytes: Vec<u8> = rout.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_x = gpu.upload_raw(&x_bytes, &[n]).expect("upload x");
    let d_rin = gpu.upload_raw(&rin_bytes, &[n]).expect("upload rin");
    let d_rout = gpu.upload_raw(&rout_bytes, &[n]).expect("upload rout");
    let d_out_in = gpu.alloc_tensor(&[n], DType::F16).expect("alloc out_in");
    let d_out_out = gpu.alloc_tensor(&[n], DType::F16).expect("alloc out_out");

    // Warm up: first call JIT-compiles the kernel (or loads it from the
    // on-disk cache), which must not be counted.
    for _ in 0..8 {
        gpu.escha_h128("escha_h128_in", &d_x, &d_rin, &d_out_in)
            .expect("warmup in");
        gpu.escha_h128("escha_h128_out", &d_x, &d_rout, &d_out_out)
            .expect("warmup out");
    }
    gpu.hip.device_synchronize().expect("sync after warmup");

    const REPS: u32 = 20_000;

    let start = Instant::now();
    for _ in 0..REPS {
        gpu.escha_h128("escha_h128_in", &d_x, &d_rin, &d_out_in)
            .expect("h128 in");
    }
    gpu.hip.device_synchronize().expect("sync after in loop");
    let elapsed_in = start.elapsed();

    let start = Instant::now();
    for _ in 0..REPS {
        gpu.escha_h128("escha_h128_out", &d_x, &d_rout, &d_out_out)
            .expect("h128 out");
    }
    gpu.hip.device_synchronize().expect("sync after out loop");
    let elapsed_out = start.elapsed();

    for (name, elapsed) in [
        ("escha_h128_in", elapsed_in),
        ("escha_h128_out", elapsed_out),
    ] {
        let per_launch = elapsed.as_secs_f64() / REPS as f64;
        let elems = n as f64;
        // Bytes moved per launch: two f32 reads (a, vec_in) + one f16 write.
        let bytes_per_launch = elems * (4.0 + 4.0 + 2.0);
        println!("{name} n={n}: {REPS} reps in {elapsed:?}");
        println!(
            "  {:.3} us/launch, {:.3} Gelem/s, {:.3} GB/s",
            per_launch * 1e6,
            elems / per_launch / 1e9,
            bytes_per_launch / per_launch / 1e9
        );
    }
}
