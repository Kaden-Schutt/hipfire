// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — per-dispatch cost floor, measured with a null kernel.
//
// Why this exists. The ds4 gfx1151 AR decode step issues 2344 kernel launches,
// of which 1704 belong to 26 "small" kernels totalling 6.559 ms — 19.4% of a
// 35.6 ms token — at an average of 3.85 us per launch. 92% of that mass runs
// below one occupancy fill (1280 wave slots), and a whole family runs exactly
// 1 or 8 waves on every dispatch.
//
// Before re-gridding or fusing any of it we need to know how much of that
// 3.85 us is even reachable. A 4096-element rmsnorm moves 16 KB; at the
// measured 207 GB/s DRAM ceiling that is 0.08 us of memory time, yet the
// kernel costs 5.15 us. If the per-dispatch floor on this part is ~2 us then
// roughly 3.4 ms of the 6.559 is irreducible dispatch cost and the lever is
// FUSION (fewer launches); if the floor is ~0.5 us then the time is really
// being spent in the kernels and the lever is GRID SHAPE.
//
// Reports both:
//   pipelined  — N launches queued back-to-back, one sync at the end. This is
//                what a decode step actually experiences.
//   serialized — sync after every launch. Full round-trip; the gap between
//                the two is how much the queue is hiding.

use rdna_compute::DType;
use rdna_compute::Gpu;
use std::time::Instant;

const SRC: &str = r#"
#include <hip/hip_runtime.h>

// Empty body: measures dispatch cost alone. The never-taken store keeps the
// pointer argument live so the signature (and kernarg setup) matches a real
// kernel rather than being elided.
extern "C" __global__ void floor_null(float* __restrict__ p) {
    if (p == (float*)1) p[0] = 0.0f;
}

// Minimal real work: one f32 read-modify-write per thread. Models the shape of
// the starved family (rmsnorm/rope/silu over a single token's hidden vector).
extern "C" __global__ void floor_touch(float* __restrict__ p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 1.000001f;
}
"#;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== per-dispatch floor: {} ===", gpu.arch);

    gpu.ensure_kernel_public("dispatch_floor", SRC, "floor_null")
        .expect("compile floor_null");
    gpu.ensure_kernel_public("dispatch_floor", SRC, "floor_touch")
        .expect("compile floor_touch");

    // 8 MB scratch: big enough for every grid below, small enough to stay
    // MALL-resident so DRAM latency never enters the measurement.
    let n = 2 * 1024 * 1024usize;
    let buf = gpu.alloc_tensor(&[n], DType::F32).expect("alloc");
    let ptr = buf.buf.as_ptr();

    // (label, workgroups, threads/wg) — waves = wgs * threads/32
    let configs: &[(&str, u32, u32)] = &[
        ("1 wg x  32t    (1 wave,    0.001 fill)", 1, 32),
        ("1 wg x 256t    (8 waves,   0.006 fill)", 1, 256),
        ("24 wg x 256t   (192 waves, 0.150 fill)", 24, 256),
        ("64 wg x 512t   (1024 wave, 0.800 fill)", 64, 512),
        ("160 wg x 256t  (1280 wave, 1.000 fill)", 160, 256),
        ("640 wg x 256t  (5120 wave, 4.000 fill)", 640, 256),
        ("2560 wg x 256t (20480 wave,16.00 fill)", 2560, 256),
    ];

    let warm = 500usize;
    let iters = 5000usize;

    for (kernel, has_n) in [("floor_null", false), ("floor_touch", true)] {
        eprintln!("\n--- {kernel} ---");
        eprintln!(
            "  {:<38} {:>12} {:>12} {:>10}",
            "grid", "pipelined us", "serial us", "queue hides"
        );
        for (label, wgs, thr) in configs {
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(ptr);
            if has_n {
                blob.push_i32((wgs * thr) as i32); // n = total work-items
            }
            // HIP launch grid is in WORKGROUPS, not work-items.
            let grid = [*wgs, 1, 1];
            let block = [*thr, 1, 1];

            let mut launch = |g: &Gpu, k: &mut hip_bridge::KernargBlob| {
                g.launch_kernel_blob(kernel, grid, block, 0, k.as_mut_slice())
                    .expect("launch")
            };

            for _ in 0..warm {
                launch(&gpu, &mut blob);
            }
            gpu.hip.device_synchronize().unwrap();

            let t = Instant::now();
            for _ in 0..iters {
                launch(&gpu, &mut blob);
            }
            gpu.hip.device_synchronize().unwrap();
            let pipelined = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let ser_iters = 1000usize;
            let t = Instant::now();
            for _ in 0..ser_iters {
                launch(&gpu, &mut blob);
                gpu.hip.device_synchronize().unwrap();
            }
            let serial = t.elapsed().as_secs_f64() * 1e6 / ser_iters as f64;

            eprintln!(
                "  {:<38} {:>12.3} {:>12.3} {:>9.2}x",
                label,
                pipelined,
                serial,
                serial / pipelined
            );
        }
    }

    eprintln!("\n  ds4 AR decode issues 1704 small-kernel launches/step totalling 6.559 ms");
    eprintln!("  (3.85 us/launch). Multiply the pipelined floor by 1704 to get the");
    eprintln!("  irreducible share; the remainder is what fusion or re-gridding can reach.");
}
