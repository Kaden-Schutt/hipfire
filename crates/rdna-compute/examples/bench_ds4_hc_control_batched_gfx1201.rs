// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — production-shape gfx1201 batched HC control screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const N_CTRL: usize = 24;
const X_DIM: usize = 16_384;
const BATCH: usize = 1_024;
const REPEATS: usize = 8;

fn upload_f16(gpu: &Gpu, values: &[u16], shape: &[usize]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let mut tensor = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload f16");
    tensor.shape = shape.to_vec();
    tensor.dtype = DType::F16;
    tensor
}

fn event_us<F>(gpu: &mut Gpu, mut launch: F) -> f64
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..REPEATS {
        launch(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("synchronize stop");
    let us =
        gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1_000.0 / REPEATS as f64;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    us
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    let x_host = (0..BATCH * X_DIM)
        .map(|i| (((i * 17 + i / 31 * 13) % 509) as f32 - 254.0) * 0.000_488_281_25)
        .collect::<Vec<_>>();
    let w_bits = (0..N_CTRL * X_DIM)
        .map(|i| {
            let magnitude = [0x2000_u16, 0x2400, 0x2800, 0x2c00][i & 3];
            magnitude
                | if (i / 7 + i / X_DIM) & 1 == 0 {
                    0
                } else {
                    0x8000
                }
        })
        .collect::<Vec<_>>();
    let base_bits = (0..N_CTRL)
        .map(|i| [0xb400_u16, 0x0000, 0x3400, 0x3800][i & 3])
        .collect::<Vec<_>>();
    let x = gpu.upload_f32(&x_host, &[BATCH, X_DIM]).expect("upload x");
    let w = upload_f16(&gpu, &w_bits, &[N_CTRL, X_DIM]);
    let base = upload_f16(&gpu, &base_bits, &[N_CTRL]);
    let inv = gpu.zeros(&[BATCH], DType::F32).expect("inv scratch");
    let reference = gpu.zeros(&[BATCH, N_CTRL], DType::F32).expect("reference");
    let fused = gpu.zeros(&[BATCH, N_CTRL], DType::F32).expect("fused24");
    let candidates = [
        gpu.zeros(&[BATCH, N_CTRL], DType::F32).expect("B1"),
        gpu.zeros(&[BATCH, N_CTRL], DType::F32).expect("B2"),
        gpu.zeros(&[BATCH, N_CTRL], DType::F32).expect("B4"),
    ];

    gpu.hc_compute_control_batched(
        &x,
        &w,
        &base,
        &reference,
        N_CTRL as i32,
        X_DIM as i32,
        BATCH as i32,
    )
    .expect("baseline");
    gpu.hc_compute_control_batched_fused24_gfx1201(
        &x,
        &w,
        &base,
        &fused,
        N_CTRL as i32,
        X_DIM as i32,
        BATCH as i32,
    )
    .expect("fused24");
    for (index, tiles) in [1usize, 2, 4].into_iter().enumerate() {
        gpu.hc_compute_control_wmma_batched_gfx1201(
            &x,
            &w,
            &base,
            &inv,
            &candidates[index],
            N_CTRL as i32,
            X_DIM as i32,
            BATCH as i32,
            tiles,
        )
        .expect("candidate");
    }
    gpu.hip.device_synchronize().expect("parity sync");
    let expected = gpu.download_f32(&reference).expect("download reference");
    let fused_actual = gpu.download_f32(&fused).expect("download fused24");
    let fused_mismatches = expected
        .iter()
        .zip(&fused_actual)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "fused24 raw_mismatches={fused_mismatches}/{}",
        expected.len()
    );
    for (index, tiles) in [1usize, 2, 4].into_iter().enumerate() {
        let actual = gpu
            .download_f32(&candidates[index])
            .expect("download candidate");
        let mut err2 = 0.0f64;
        let mut ref2 = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&a, &b) in expected.iter().zip(&actual) {
            let delta = (a - b).abs();
            max_abs = max_abs.max(delta);
            err2 += (delta as f64) * (delta as f64);
            ref2 += (a as f64) * (a as f64);
        }
        println!(
            "B{tiles} rel_rmse={:.8} max_abs={max_abs:.8}",
            (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt()
        );
    }

    for _ in 0..3 {
        gpu.hc_compute_control_batched(
            &x,
            &w,
            &base,
            &reference,
            N_CTRL as i32,
            X_DIM as i32,
            BATCH as i32,
        )
        .expect("warm baseline");
    }
    let baseline_us = event_us(&mut gpu, |gpu| {
        gpu.hc_compute_control_batched(
            &x,
            &w,
            &base,
            &reference,
            N_CTRL as i32,
            X_DIM as i32,
            BATCH as i32,
        )
        .expect("time baseline")
    });
    let fused_us = event_us(&mut gpu, |gpu| {
        gpu.hc_compute_control_batched_fused24_gfx1201(
            &x,
            &w,
            &base,
            &fused,
            N_CTRL as i32,
            X_DIM as i32,
            BATCH as i32,
        )
        .expect("time fused24")
    });
    println!(
        "fused24 baseline_us={baseline_us:.3} candidate_us={fused_us:.3} speedup_x={:.3}",
        baseline_us / fused_us
    );
    for (index, tiles) in [1usize, 2, 4].into_iter().enumerate() {
        let candidate_us = event_us(&mut gpu, |gpu| {
            gpu.hc_compute_control_wmma_batched_gfx1201(
                &x,
                &w,
                &base,
                &inv,
                &candidates[index],
                N_CTRL as i32,
                X_DIM as i32,
                BATCH as i32,
                tiles,
            )
            .expect("time candidate")
        });
        println!(
            "B{tiles} baseline_us={baseline_us:.3} candidate_us={candidate_us:.3} speedup_x={:.3}",
            baseline_us / candidate_us
        );
    }
}
