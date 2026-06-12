// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Microbench for DeltaNet tree conv1d_silu_split.
//!
//! On gfx1151, unset `HIPFIRE_CONV1D_TREE_GFX1151` measures the token-parallel
//! gfx1151 route. Set `HIPFIRE_CONV1D_TREE_GFX1151=0` before process startup to
//! measure the generic channel-parallel tree kernel.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("bench_conv1d_tree_gfx1151 requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::{DType, Gpu};

    let mut gpu = Gpu::init().expect("GPU init failed");
    let route = if std::env::var("HIPFIRE_CONV1D_TREE_GFX1151").as_deref() == Ok("0") {
        "generic"
    } else {
        "gfx1151"
    };

    let k_dim = 128usize;
    let v_dim = 256usize;
    let n_ch = 2 * k_dim + v_dim;
    let trials: usize = std::env::var("TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let weight: Vec<f32> = (0..n_ch * 4)
        .map(|i| (((i * 7919 + 17) % 127) as f32 - 63.0) * 0.00625)
        .collect();
    let state: Vec<f32> = (0..n_ch * 3)
        .map(|i| (((i * 2027 + 31) % 113) as f32 - 56.0) * 0.0078125)
        .collect();
    let w = gpu.upload_f32(&weight, &[n_ch, 4]).unwrap();
    let s = gpu.upload_f32(&state, &[n_ch, 3]).unwrap();

    println!("route={route} arch={} trials={trials}", gpu.arch);
    for &n_tokens in &[4usize, 8, 16, 32, 64] {
        let input: Vec<f32> = (0..n_tokens * n_ch)
            .map(|i| (((i * 104_729 + 19) % 257) as f32 - 128.0) * 0.004)
            .collect();
        let parents: Vec<i32> = (0..n_tokens as i32)
            .map(|t| if t == 0 { -1 } else { (t - 1) / 2 })
            .collect();

        let x = gpu.upload_f32(&input, &[n_tokens, n_ch]).unwrap();
        let p = alloc_i32(&mut gpu, &parents);
        let q = gpu.zeros(&[n_tokens, k_dim], DType::F32).unwrap();
        let k = gpu.zeros(&[n_tokens, k_dim], DType::F32).unwrap();
        let v = gpu.zeros(&[n_tokens, v_dim], DType::F32).unwrap();

        let us = time_us(&mut gpu, trials, |gpu| {
            gpu.conv1d_silu_split_tree_f32_n(&q, &k, &v, &x, &w, &s, &p, k_dim, v_dim, n_tokens)
                .expect("tree conv");
        });
        println!("n_tokens={n_tokens:>2} medianish_us={us:.3}");

        gpu.free_tensor(x).unwrap();
        gpu.free_tensor(p).unwrap();
        gpu.free_tensor(q).unwrap();
        gpu.free_tensor(k).unwrap();
        gpu.free_tensor(v).unwrap();
    }
}

#[cfg(feature = "deltanet")]
fn alloc_i32(gpu: &mut rdna_compute::Gpu, data: &[i32]) -> rdna_compute::GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len() * 4], rdna_compute::DType::Raw)
        .unwrap();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}

#[cfg(feature = "deltanet")]
fn time_us(
    gpu: &mut rdna_compute::Gpu,
    iters: usize,
    mut f: impl FnMut(&mut rdna_compute::Gpu),
) -> f32 {
    for _ in 0..16 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..iters {
        f(gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    ms * 1000.0 / iters as f32
}
