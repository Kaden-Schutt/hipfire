// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throughput bench for the Mamba-2 SSD bf16-WMMA chunked prefill at production
//! Nano-4B scale (H=96, head_dim=80, state=128, n_groups=8). hipEvent-timed,
//! µs/call across a sweep of prompt lengths. Baseline for the per-group CB-Gram
//! reuse optimization.
//!
//!   hipfire lock acquire
//!   cargo run --release -p hipfire-arch-nemotron --example bench_ssd_chunk_wmma

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch = {}", gpu.arch);

    let (h, dh, n, g) = (96usize, 80usize, 128usize, 8usize);
    let xd = h * dh;
    let bd = g * n;
    let chunk = 16usize;

    let mut seed = 0xC0FFEE11u32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };

    let alog = gpu
        .upload_f32(&(0..h).map(|_| rng() * 0.5).collect::<Vec<_>>(), &[h])
        .unwrap();
    let dd = gpu
        .upload_f32(&(0..h).map(|_| rng()).collect::<Vec<_>>(), &[h])
        .unwrap();
    let dtb = gpu
        .upload_f32(&(0..h).map(|_| rng()).collect::<Vec<_>>(), &[h])
        .unwrap();

    println!(
        "{:>7} {:>12} {:>12} {:>10}",
        "seq", "wmma us", "seq_f32 us", "speedup"
    );
    for &seq in &[128usize, 256, 512, 1024] {
        let x = gpu
            .upload_f32(
                &(0..seq * xd).map(|_| rng()).collect::<Vec<_>>(),
                &[seq * xd],
            )
            .unwrap();
        let b = gpu
            .upload_f32(
                &(0..seq * bd).map(|_| rng()).collect::<Vec<_>>(),
                &[seq * bd],
            )
            .unwrap();
        let c = gpu
            .upload_f32(
                &(0..seq * bd).map(|_| rng()).collect::<Vec<_>>(),
                &[seq * bd],
            )
            .unwrap();
        let dt = gpu
            .upload_f32(&(0..seq * h).map(|_| rng()).collect::<Vec<_>>(), &[seq * h])
            .unwrap();
        let y = gpu.zeros(&[seq * xd], DType::F32).unwrap();
        let st = gpu.zeros(&[h * dh * n], DType::F32).unwrap();

        const ITERS: usize = 100;
        let time = |gpu: &mut Gpu, wmma: bool| -> f64 {
            let once = |gpu: &mut Gpu| {
                if wmma {
                    gpu.mamba2_ssd_chunk_wmma(
                        &y,
                        &st,
                        &x,
                        &b,
                        &c,
                        &dt,
                        &alog,
                        &dd,
                        &dtb,
                        seq,
                        h,
                        dh,
                        n,
                        g,
                        0.0,
                        f32::INFINITY,
                        chunk,
                    )
                    .unwrap();
                } else {
                    gpu.mamba2_ssd_seq_f32(
                        &y,
                        &st,
                        &x,
                        &b,
                        &c,
                        &dt,
                        &alog,
                        &dd,
                        &dtb,
                        seq,
                        h,
                        dh,
                        n,
                        g,
                        0.0,
                        f32::INFINITY,
                    )
                    .unwrap();
                }
            };
            for _ in 0..10 {
                once(gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            let ev0 = gpu.hip.event_create().unwrap();
            let ev1 = gpu.hip.event_create().unwrap();
            gpu.hip.event_record(&ev0, None).unwrap();
            for _ in 0..ITERS {
                once(gpu);
            }
            gpu.hip.event_record(&ev1, None).unwrap();
            gpu.hip.event_synchronize(&ev1).unwrap();
            gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1000.0 / ITERS as f64
        };
        let wmma_us = time(&mut gpu, true);
        let seq_us = time(&mut gpu, false);
        println!(
            "{seq:>7} {wmma_us:>12.1} {seq_us:>12.1} {:>10.2}",
            seq_us / wmma_us
        );
    }
}
