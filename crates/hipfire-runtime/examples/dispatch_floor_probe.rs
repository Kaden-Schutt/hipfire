// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Measures the per-launch HIP dispatch floor carried by every in-process
//! kernel timing in this tree, so microbenchmark numbers can be corrected for
//! instrument error instead of being reported as kernel time.
//!
//! Why this exists: `redline-pub/docs/DISPATCH-FLOOR.md` measures HIP at
//! ~2.13 us/dispatch and retained PM4 at 0.176-0.211 us/dispatch on RDNA
//! (ROCm 7.2-era hosts). Any single-kernel A/B run through `launch_maybe_blob`
//! therefore carries a constant additive term of roughly that size. At a 34 us
//! kernel that is ~6%, and the MQ4-GL-vs-uniform gap under investigation is
//! only ~2.9 us — comparable to ONE dispatch. Reporting such a gap without
//! bounding the floor is reporting the instrument.
//!
//! Method: launch the real qt=40 GEMV at a shape whose arithmetic is
//! negligible (m=1, k=256 reads 130 B of weights and 1 KiB of x), through the
//! exact same wrapper and `launch_maybe_blob` path the benchmarks use. The
//! measured time is dominated by submission + fence, not by the kernel. Then
//! scale m upward on the same k to separate the constant term from the
//! per-byte term by regression, which does not depend on the null-kernel
//! assumption holding exactly.
//!
//! Run:
//!   cargo build --release -p hipfire-runtime --example dispatch_floor_probe
//!   ./target/release/examples/dispatch_floor_probe

use half::f16;
use rdna_compute::{Gpu, GL_CB4};
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 200;

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("arch={}", gpu.arch);
    eprintln!(
        "reference (redline-pub docs/DISPATCH-FLOOR.md, ROCm 7.2-era RDNA):\n  \
         hipGraphLaunch ~2.13 us/disp | Redline BoundarySerialized ~1.30 | \
         PM4 IB conservative 0.176-0.211 | PM4 IB aggressive 0.124-0.148"
    );

    // k fixed so the per-group decode path is identical across rows; only the
    // row count (and thus the byte count) varies. gpr = 20 at k = 5120.
    // DPM confound: a sweep of only tiny shapes never loads the GPU, so clocks
    // stay low and the apparent fixed cost is inflated. An earlier version of
    // this probe swept m=1..128 and fitted a 31.9 us "floor" that cannot be
    // real -- subtracting it from the uniform arm implied 3.6 TB/s on a ~640
    // GB/s part. Sweep shapes that all keep the device busy instead, and let
    // the regression separate the constant from the per-byte term.
    let k = 5120usize;
    let rows = [512usize, 1024, 2048, 3072, 4096, 6144, 8192, 12288];
    println!(
        "\n{:>7}  {:>11}  {:>10}  {:>10}  {:>10}  {:>9}",
        "m", "bytes", "host_us", "gpu_us", "overhead", "gpu_GB/s"
    );
    let mut samples: Vec<(f64, f64)> = Vec::new(); // (bytes, gpu seconds)
    for &m in &rows {
        let (blob, x) = build(m, k);
        let (hmed, gmed) = time_gemv(&mut gpu, &blob, &x, m, k);
        let bytes = (m * (k / 256) * 130) as f64;
        samples.push((bytes, gmed));
        println!(
            "{:>7}  {:>11}  {:>10.3}  {:>10.3}  {:>10.3}  {:>9.1}",
            m,
            bytes as u64,
            hmed * 1e6,
            gmed * 1e6,
            (hmed - gmed) * 1e6,
            bytes / gmed / 1e9
        );
    }

    // Least-squares fit t = a + b*bytes. `a` is the launch-invariant term:
    // submission, fence, and any fixed setup. `b` is effective inverse
    // bandwidth. Reported with r^2 so a bad fit is visible rather than silent.
    let n = samples.len() as f64;
    let sx: f64 = samples.iter().map(|s| s.0).sum();
    let sy: f64 = samples.iter().map(|s| s.1).sum();
    let sxx: f64 = samples.iter().map(|s| s.0 * s.0).sum();
    let sxy: f64 = samples.iter().map(|s| s.0 * s.1).sum();
    let denom = n * sxx - sx * sx;
    let b = (n * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / n;
    let ybar = sy / n;
    let ss_tot: f64 = samples.iter().map(|s| (s.1 - ybar).powi(2)).sum();
    let ss_res: f64 = samples.iter().map(|s| (s.1 - (a + b * s.0)).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    println!("\nfit  t = a + b*bytes   (r^2 = {r2:.5})");
    println!("  a (launch-invariant) = {:.3} us", a * 1e6);
    println!("  1/b (effective BW)   = {:.1} GB/s", 1.0 / b / 1e9);

    // What that constant does to the numbers this session reported.
    println!("\ncorrection applied to this session's m=4096 k=5120 medians:");
    let floor_us = a * 1e6;
    for (label, measured_us) in [
        ("uniform HFQ4G256 multirow", 34.32f64),
        ("MQ4-GL multirow r4", 37.21),
        ("MQ4-GL single-row", 48.20),
        ("MQ4-Lloyd (branch-fixed)", 48.57),
    ] {
        let corrected = measured_us - floor_us;
        println!(
            "  {label:26}  measured {measured_us:6.2} us  ->  kernel {corrected:6.2} us  \
             (floor = {:.1}% of measured)",
            100.0 * floor_us / measured_us
        );
    }
    let g = 37.21 - floor_us;
    let u = 34.32 - floor_us;
    println!(
        "\n  GL-vs-uniform gap: {:.2} us absolute, {:.1}% of measured vs {:.1}% of kernel time",
        37.21 - 34.32,
        100.0 * (37.21 - 34.32) / 34.32,
        100.0 * (g - u) / u
    );
    println!(
        "  gap expressed in dispatches: {:.2}x one HIP dispatch (2.13 us)",
        (37.21 - 34.32) / 2.13
    );
}

/// Pack a qt=40 blob and a matching x. Values are irrelevant here — only the
/// byte count and the executed code path matter — so this uses a cheap
/// deterministic fill rather than the parity harness's Gaussian construction.
fn build(m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 128;
    let mut blob = vec![0u8; idx_bytes + m * gpr * 2];
    for (i, b) in blob[..idx_bytes].iter_mut().enumerate() {
        *b = ((i * 37) & 0xFF) as u8;
    }
    let scale = f16::from_f32(0.05).to_bits();
    for g in 0..(m * gpr) {
        blob[idx_bytes + g * 2] = (scale & 0xFF) as u8;
        blob[idx_bytes + g * 2 + 1] = (scale >> 8) as u8;
    }
    let x = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    (blob, x)
}

/// Returns (host_median_s, gpu_median_s). Host time is `Instant` around
/// launch+`device_synchronize`, i.e. what every benchmark in this tree
/// reports. GPU time comes from the in-tree HIP-event profiler that the gemv
/// wrapper already feeds via `profile::begin_timer`. The difference between
/// them is launch + fence + sync round-trip, measured per shape rather than
/// inferred from a regression intercept.
fn time_gemv(gpu: &mut Gpu, blob: &[u8], x: &[f32], m: usize, k: usize) -> (f64, f64) {
    let d_a = gpu.upload_raw(blob, &[blob.len()]).expect("upload blob");
    let d_x = gpu.upload_f32(x, &[k]).expect("upload x");
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).expect("alloc y");

    for _ in 0..WARMUP {
        gpu.gemv_mq4g256gl(&d_a, &d_x, &d_y, m, k).expect("gemv");
    }
    gpu.hip.device_synchronize().expect("sync");

    let mut host = Vec::with_capacity(ITERS);
    rdna_compute::profile::start();
    for _ in 0..ITERS {
        gpu.hip.device_synchronize().expect("sync");
        let t0 = Instant::now();
        gpu.gemv_mq4g256gl(&d_a, &d_x, &d_y, m, k).expect("gemv");
        gpu.hip.device_synchronize().expect("sync");
        host.push(t0.elapsed().as_secs_f64());
    }
    let entries = rdna_compute::profile::stop().unwrap_or_default();

    let mut gpu_us: Vec<f64> = entries
        .iter()
        .filter(|e| e.kernel.contains("mq4g256gl"))
        .map(|e| e.time_us)
        .collect();

    host.sort_by(|p, q| p.partial_cmp(q).unwrap());
    gpu_us.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let hmed = host[host.len() / 2];
    let gmed = if gpu_us.is_empty() {
        f64::NAN
    } else {
        gpu_us[gpu_us.len() / 2] / 1e6
    };
    (hmed, gmed)
}

#[allow(dead_code)]
fn cb() -> [f32; 16] {
    GL_CB4
}
