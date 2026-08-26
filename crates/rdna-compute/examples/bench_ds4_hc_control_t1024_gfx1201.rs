// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — DS4 gfx1201 wide HC control/finalize channel screen.

use rdna_compute::{DType, Gpu, GpuTensor};

const N_CTRL: usize = 24;
const X_DIM: usize = 16_384;
const EPS: f32 = 1.0e-6;
const POST_SCALE: f32 = 1.5;
const SINKHORN_ITERS: i32 = 20;
const REPEATS: usize = 256;
const TRIALS: usize = 9;

fn upload_f16(gpu: &Gpu, values: &[u16], shape: &[usize]) -> GpuTensor {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) };
    let mut tensor = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload f16");
    tensor.shape = shape.to_vec();
    tensor.dtype = DType::F16;
    tensor
}

fn launch(
    gpu: &mut Gpu,
    streams: &GpuTensor,
    weights: &GpuTensor,
    base: &GpuTensor,
    output: &GpuTensor,
    alpha: &GpuTensor,
    wide: bool,
) {
    gpu.hc_compute_control_vec4_finalize(
        streams,
        weights,
        base,
        output,
        alpha,
        N_CTRL as i32,
        X_DIM as i32,
        EPS,
        POST_SCALE,
        SINKHORN_ITERS,
        true,
        wide,
    )
    .expect("HC control/finalize");
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

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn ordered_bits(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        (bits | 0x8000_0000) as i64
    } else {
        (!bits) as i64
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");

    let streams_host = (0..X_DIM)
        .map(|index| {
            let value = ((index * 17 + index / 31 * 13) % 509) as f32 - 254.0;
            value * 0.000_488_281_25
        })
        .collect::<Vec<_>>();
    let weight_bits = (0..N_CTRL * X_DIM)
        .map(|index| {
            let magnitude = [0x2000_u16, 0x2400, 0x2800, 0x2c00][index & 3];
            magnitude
                | if (index / 7 + index / X_DIM) & 1 == 0 {
                    0
                } else {
                    0x8000
                }
        })
        .collect::<Vec<_>>();
    let base_bits = (0..N_CTRL)
        .map(|index| [0xb400_u16, 0x0000, 0x3400, 0x3800][index & 3])
        .collect::<Vec<_>>();
    let alpha_bits = [0x3000_u16, 0x3800, 0x3b00];

    let streams = gpu
        .upload_f32(&streams_host, &[X_DIM])
        .expect("upload streams");
    let weights = upload_f16(&gpu, &weight_bits, &[N_CTRL, X_DIM]);
    let base = upload_f16(&gpu, &base_bits, &[N_CTRL]);
    let alpha = upload_f16(&gpu, &alpha_bits, &[3]);
    // rsqrt_once uses [24]=rsqrt, [25]=ready, [26]=ticket.
    let incumbent = gpu.zeros(&[27], DType::F32).expect("incumbent output");
    let candidate = gpu.zeros(&[27], DType::F32).expect("candidate output");

    launch(
        &mut gpu, &streams, &weights, &base, &incumbent, &alpha, false,
    );
    launch(
        &mut gpu, &streams, &weights, &base, &candidate, &alpha, true,
    );
    gpu.hip.device_synchronize().expect("parity synchronize");
    let reference = gpu.download_f32(&incumbent).expect("download incumbent");
    let actual = gpu.download_f32(&candidate).expect("download candidate");
    let mut raw_equal = 0usize;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_ulp = 0_i64;
    for (&expected, &candidate) in reference[..N_CTRL].iter().zip(&actual[..N_CTRL]) {
        raw_equal += usize::from(expected.to_bits() == candidate.to_bits());
        let abs = (expected - candidate).abs();
        max_abs = max_abs.max(abs);
        if expected.abs() > 1.0e-12 {
            max_rel = max_rel.max(abs / expected.abs());
        }
        max_ulp = max_ulp.max((ordered_bits(expected) - ordered_bits(candidate)).abs());
    }
    assert_eq!(reference[25].to_bits(), 0, "incumbent ready not reset");
    assert_eq!(reference[26].to_bits(), 0, "incumbent ticket not reset");
    assert_eq!(actual[25].to_bits(), 0, "candidate ready not reset");
    assert_eq!(actual[26].to_bits(), 0, "candidate ticket not reset");

    for _ in 0..20 {
        launch(
            &mut gpu, &streams, &weights, &base, &incumbent, &alpha, false,
        );
        launch(
            &mut gpu, &streams, &weights, &base, &candidate, &alpha, true,
        );
    }
    gpu.hip.device_synchronize().expect("warmup synchronize");
    let mut incumbent_us = Vec::with_capacity(TRIALS);
    let mut candidate_us = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        if trial & 1 == 0 {
            incumbent_us.push(event_us(&mut gpu, |gpu| {
                launch(gpu, &streams, &weights, &base, &incumbent, &alpha, false)
            }));
            candidate_us.push(event_us(&mut gpu, |gpu| {
                launch(gpu, &streams, &weights, &base, &candidate, &alpha, true)
            }));
        } else {
            candidate_us.push(event_us(&mut gpu, |gpu| {
                launch(gpu, &streams, &weights, &base, &candidate, &alpha, true)
            }));
            incumbent_us.push(event_us(&mut gpu, |gpu| {
                launch(gpu, &streams, &weights, &base, &incumbent, &alpha, false)
            }));
        }
    }
    let incumbent_us = median(&mut incumbent_us);
    let candidate_us = median(&mut candidate_us);
    println!(
        "raw_equal={raw_equal}/{N_CTRL} max_abs={max_abs:.9e} max_rel={max_rel:.9e} max_ulp={max_ulp} incumbent_us={incumbent_us:.6} candidate_us={candidate_us:.6} speedup_x={:.4} saved_us_per_call={:.6} saved_ms_per_rank_token={:.6}",
        incumbent_us / candidate_us,
        incumbent_us - candidate_us,
        (incumbent_us - candidate_us) * 86.0 / 1_000.0,
    );
}
