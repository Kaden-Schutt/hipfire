// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — DS4 gfx1201 wide YaRN RoPE decode channel screen.

use rdna_compute::{Gpu, GpuTensor};

const HEAD_DIM: usize = 512;
const ROT: i32 = 64;
const REPEATS: usize = 2_000;
const TRIALS: usize = 9;

fn upload_i32(gpu: &mut Gpu, value: i32) -> GpuTensor {
    gpu.upload_raw(&value.to_le_bytes(), &[1])
        .expect("upload i32")
}

fn input(elements: usize, salt: u64) -> Vec<f32> {
    let mut state = 0x1201_0731_D54A_0001_u64 ^ salt;
    (0..elements)
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

#[derive(Clone, Copy)]
struct RopeParams {
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    corr_low: f32,
    corr_high: f32,
}

const COMPRESSED: RopeParams = RopeParams {
    freq_base: 160_000.0,
    freq_scale: 0.0625,
    ext_factor: 1.0,
    attn_factor: 0.782_966_4,
    corr_low: 10.0,
    corr_high: 30.0,
};

#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &mut Gpu,
    wide: bool,
    q: &GpuTensor,
    k: &GpuTensor,
    pos: &GpuTensor,
    heads_q: usize,
    heads_k: usize,
    inverse: i32,
) {
    gpu.rope_tail_yarn_interleaved(
        wide,
        q,
        k,
        pos,
        heads_q as i32,
        heads_k as i32,
        HEAD_DIM as i32,
        ROT,
        COMPRESSED.freq_base,
        COMPRESSED.freq_scale,
        COMPRESSED.ext_factor,
        COMPRESSED.attn_factor,
        COMPRESSED.corr_low,
        COMPRESSED.corr_high,
        inverse,
    )
    .expect("YaRN RoPE launch");
}

fn event_us<F>(gpu: &mut Gpu, mut f: F) -> f64
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..REPEATS {
        f(gpu);
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

fn screen(gpu: &mut Gpu, heads_q: usize, heads_k: usize, inverse: i32) -> (f64, f64, usize) {
    let q_host = input(heads_q * HEAD_DIM, (heads_q as u64) << 32 | inverse as u64);
    let k_elements = heads_k.max(1) * HEAD_DIM;
    let k_host = input(k_elements, (heads_k as u64) << 16 | 0xBEEF);
    let pos = upload_i32(gpu, 2_052);

    let q_ref = gpu
        .upload_f32(&q_host, &[heads_q, HEAD_DIM])
        .expect("q ref");
    let q_wide = gpu
        .upload_f32(&q_host, &[heads_q, HEAD_DIM])
        .expect("q wide");
    let k_ref = gpu.upload_f32(&k_host, &[k_elements]).expect("k ref");
    let k_wide = gpu.upload_f32(&k_host, &[k_elements]).expect("k wide");

    launch(gpu, false, &q_ref, &k_ref, &pos, heads_q, heads_k, inverse);
    launch(gpu, true, &q_wide, &k_wide, &pos, heads_q, heads_k, inverse);
    gpu.hip.device_synchronize().expect("parity synchronize");
    let q_expected = gpu.download_f32(&q_ref).expect("download q ref");
    let q_actual = gpu.download_f32(&q_wide).expect("download q wide");
    let k_expected = gpu.download_f32(&k_ref).expect("download k ref");
    let k_actual = gpu.download_f32(&k_wide).expect("download k wide");
    let mismatches = q_expected
        .iter()
        .zip(&q_actual)
        .chain(k_expected.iter().zip(&k_actual))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    for _ in 0..20 {
        launch(gpu, false, &q_ref, &k_ref, &pos, heads_q, heads_k, inverse);
        launch(gpu, true, &q_wide, &k_wide, &pos, heads_q, heads_k, inverse);
    }
    gpu.hip.device_synchronize().expect("warmup synchronize");

    let mut reference = Vec::with_capacity(TRIALS);
    let mut candidate = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        if trial & 1 == 0 {
            reference.push(event_us(gpu, |gpu| {
                launch(gpu, false, &q_ref, &k_ref, &pos, heads_q, heads_k, inverse)
            }));
            candidate.push(event_us(gpu, |gpu| {
                launch(gpu, true, &q_wide, &k_wide, &pos, heads_q, heads_k, inverse)
            }));
        } else {
            candidate.push(event_us(gpu, |gpu| {
                launch(gpu, true, &q_wide, &k_wide, &pos, heads_q, heads_k, inverse)
            }));
            reference.push(event_us(gpu, |gpu| {
                launch(gpu, false, &q_ref, &k_ref, &pos, heads_q, heads_k, inverse)
            }));
        }
    }
    (median(&mut reference), median(&mut candidate), mismatches)
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    for heads_q in [24, 16] {
        for (heads_k, inverse) in [(1, 0), (0, 1)] {
            let (reference_us, wide_us, mismatches) = screen(&mut gpu, heads_q, heads_k, inverse);
            println!(
                "heads_q={heads_q} heads_k={heads_k} inverse={inverse} reference_us={reference_us:.6} wide_us={wide_us:.6} speedup_x={:.4} saved_us={:.6} raw_mismatches={mismatches}",
                reference_us / wide_us,
                reference_us - wide_us,
            );
        }
    }
}
