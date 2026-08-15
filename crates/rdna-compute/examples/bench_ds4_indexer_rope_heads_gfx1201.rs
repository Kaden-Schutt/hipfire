// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — DS4 gfx1201 indexer-Q head-parallel RoPE channel screen.

use rdna_compute::{Gpu, GpuTensor};

const HEADS: usize = 64;
const HEAD_DIM: usize = 128;
const ROT: i32 = 64;
const FREQ_BASE: f32 = 10_000.0;
const REPEATS: usize = 2_000;
const TRIALS: usize = 9;

fn upload_i32(gpu: &mut Gpu, value: i32) -> GpuTensor {
    gpu.upload_raw(&value.to_le_bytes(), &[1])
        .expect("upload i32")
}

fn input(position: i32) -> Vec<f32> {
    let mut state = 0x1201_0731_u64 ^ position as u64;
    (0..HEADS * HEAD_DIM)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            match index & 31 {
                0 => 0.0,
                1 => -0.0,
                2 => 0.125,
                3 => -0.25,
                _ => (((state >> 32) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.25,
            }
        })
        .collect()
}

fn launch_reference(gpu: &mut Gpu, q: &GpuTensor, pos: &GpuTensor) {
    gpu.rope_tail_interleaved(q, q, pos, HEADS as i32, 0, HEAD_DIM as i32, ROT, FREQ_BASE)
        .expect("reference RoPE");
}

fn launch_candidate(gpu: &mut Gpu, q: &GpuTensor, pos: &GpuTensor, head_waves: u32) {
    gpu.rope_tail_interleaved_h64d128r64_gfx1201(q, pos, FREQ_BASE, head_waves)
        .expect("candidate RoPE");
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

fn parity(gpu: &mut Gpu, position: i32, head_waves: u32) -> usize {
    let host = input(position);
    let reference = gpu
        .upload_f32(&host, &[HEADS, HEAD_DIM])
        .expect("upload reference");
    let candidate = gpu
        .upload_f32(&host, &[HEADS, HEAD_DIM])
        .expect("upload candidate");
    let pos = upload_i32(gpu, position);
    launch_reference(gpu, &reference, &pos);
    launch_candidate(gpu, &candidate, &pos, head_waves);
    gpu.hip.device_synchronize().expect("parity synchronize");
    let expected = gpu.download_f32(&reference).expect("download reference");
    let actual = gpu.download_f32(&candidate).expect("download candidate");
    for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "raw-bit mismatch position={position} head_waves={head_waves} index={index}"
        );
    }
    expected.len()
}

fn screen(gpu: &mut Gpu, head_waves: u32) {
    let host = input(2052);
    let reference = gpu
        .upload_f32(&host, &[HEADS, HEAD_DIM])
        .expect("upload reference");
    let candidate = gpu
        .upload_f32(&host, &[HEADS, HEAD_DIM])
        .expect("upload candidate");
    let pos = upload_i32(gpu, 2052);
    for _ in 0..20 {
        launch_reference(gpu, &reference, &pos);
        launch_candidate(gpu, &candidate, &pos, head_waves);
    }
    gpu.hip.device_synchronize().expect("warmup synchronize");

    let mut reference_us = Vec::with_capacity(TRIALS);
    let mut candidate_us = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        if trial & 1 == 0 {
            reference_us.push(event_us(gpu, |gpu| launch_reference(gpu, &reference, &pos)));
            candidate_us.push(event_us(gpu, |gpu| {
                launch_candidate(gpu, &candidate, &pos, head_waves)
            }));
        } else {
            candidate_us.push(event_us(gpu, |gpu| {
                launch_candidate(gpu, &candidate, &pos, head_waves)
            }));
            reference_us.push(event_us(gpu, |gpu| launch_reference(gpu, &reference, &pos)));
        }
    }
    let reference_us = median(&mut reference_us);
    let candidate_us = median(&mut candidate_us);
    println!(
        "head_waves={head_waves} reference_us={reference_us:.6} candidate_us={candidate_us:.6} speedup_x={:.4} saved_us_per_call={:.6} saved_ms_per_rank_token={:.6}",
        reference_us / candidate_us,
        reference_us - candidate_us,
        (reference_us - candidate_us) * 21.0 / 1_000.0,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    let positions = [0, 1, 2052, 131_071];
    let head_waves = [8, 16, 32];
    let mut comparisons = 0usize;
    for head_waves in head_waves {
        for position in positions {
            comparisons += parity(&mut gpu, position, head_waves);
        }
        screen(&mut gpu, head_waves);
    }
    println!("raw_bit_comparisons={comparisons}");
}
